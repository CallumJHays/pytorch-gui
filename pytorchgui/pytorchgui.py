import json
from datetime import datetime
from torch.autograd import Variable, Function
from torch.nn import Parameter
from string import ascii_lowercase
import numpy as np
import copy
from sanic import Sanic
import sanic.response as response
from sanic_cors import CORS
import webbrowser
import os
import ipywidgets
import IPython.display
from multiprocessing import Process
import atexit
import time


class Graph:
    """
    A graph structure used to analyse a pytorch graph for pytorchgui
    """

    def __init__(self, module, dataloader=None):
        self.module = module
        # dictionary of references to the modules and parameters
        self.module_tree = {}
        self.module_to_id = {}
        self.param_to_id = {}
        self.functional_graph = {}
        self.id_gen = alphabetical_ids()
        self.prev_id = None
        self.dataloader = dataloader

        unique_param_ids = set()
        for param in module.parameters():
            param_id = next(self.id_gen)
            self.prev_id = param_id
            self.param_to_id[param] = param_id
            unique_param_ids.add(param_id)

        def register_submodules(module):
            module_id = next(self.id_gen)
            self.prev_id = module_id
            self.module_to_id[module] = module_id

            self.module_tree[module_id] = {
                'children': {},
                'params': {}
            }

            for key, submodule in module._modules.items():
                submodule_id = register_submodules(submodule)
                self.module_tree[module_id]['children'][key] = submodule_id
                for key, param_id in self.module_tree[submodule_id]['params'].items():
                    unique_param_ids.remove(param_id)

            for key, param in module._parameters.items():
                param_id = self.param_to_id[param]
                if param_id in unique_param_ids:
                    self.module_tree[module_id]['params'][key] = param_id

            return module_id

        register_submodules(self.module)

    def instrumented_forward(self, *inputs):
        self.functional_graph = {}
        node_to_id = {**self.param_to_id, **self.module_to_id}
        activations = {}
        node_id_gen = alphabetical_ids(self.prev_id)

        # fills the functional graph
        def fill_functional_graph(node, parent_module):
            if node not in node_to_id:
                node_id = next(node_id_gen)
                node_to_id[node] = node_id

                self.functional_graph[node_id] = {
                    'type': type(node).__name__,
                    'dependencies': set(),
                    'parent_module': self.module_to_id[parent_module]
                }

                if hasattr(node, 'previous_functions'):
                    for dep, _ in node.previous_functions:
                        if dep not in self.functional_graph:
                            fill_functional_graph(dep, parent_module)
                        self.functional_graph[node_id]['dependencies'].add(
                            node_to_id[dep])

        def forward_hook(module, inputs, output):
            activations[self.module_to_id[module]] = {
                'inputs': tuple(np.squeeze(input.data.numpy()) for input in inputs),
                'output': np.squeeze(output.data.numpy())
            }

            fill_functional_graph(output.creator, module)

        if len(inputs) == 0:
            if self.dataloader is None:
                raise Exception(
                    "Either inputs or a dataloader must be provided")
            inputs, targets = next(self.dataloader.__iter__())
            if isinstance(inputs, tuple):
                inputs = tuple(Variable(input) for input in inputs)
            else:
                inputs = tuple([Variable(inputs)])

        for input in inputs:
            input_id = next(node_id_gen)
            node_to_id[input] = input_id
            self.functional_graph[input_id] = {
                'type': 'Input: ' + type(input.data).__name__ + ' ('
                + ", ".join(map(str, input.data.size())) + ")",
                'dependencies': set(),
                'parent_module': None
            }

        handles = [module.register_forward_hook(
            forward_hook) for module in self.module_to_id]

        res = self.module.forward(*inputs)

        forward_hook(self.module, inputs, res)

        res_id = next(node_id_gen)
        node_to_id[res] = res_id
        self.functional_graph[res_id] = {
            'type': 'Output: ' + type(res.data).__name__ + ' ('
            + ", ".join(map(str, res.data.size())) + ")",
            'dependencies': set([node_to_id[res.creator]]),
            'parent_module': self.module_to_id[self.module]
        }

        # remove the handle to keep the function stateless
        for handle in handles:
            handle.remove()

        return {
            'activations': activations,
            'functional_graph': self.functional_graph,
            'result': np.squeeze(res.data.numpy())
        }

    def serialize(self):
        serialized = {}

        for module, module_id in self.module_to_id.items():
            module_dict = self.module_tree[module_id]

            serialized[module_id] = {
                'type': 'Module',
                'subtype': type(module).__name__,
                'params': copy.copy(module_dict['params']),
                'children': copy.copy(module_dict['children'])
            }

        for param, param_id in self.param_to_id.items():
            serialized[param_id] = {
                'type': 'Parameter',
                'subtype': type(param.data).__name__,
                'shape': param.data.size()
            }

        for func_id, func_dict in self.functional_graph.items():
            serialized[func_id] = {
                'type': 'Function',
                'subtype': func_dict['type'],
                'dependencies': list(copy.copy(func_dict['dependencies'])),
                'parent_module': func_dict['parent_module']
            }

        return json.dumps(serialized)

    def serve(self, port=7060, url_base="/api/v1", jupyter_widget=True):
        """
        Serves the current graph using a sanic webserver, providing a few api
        end-points:

            `graph_spec`: Serves the result of `Graph.serialize`.

            `activations`: Runs `Graph.instrumented_forward` and serializes the
                activations of the requested `target_ids` (POST data attribute)
                as JSON to be viewable by the front-end visualizer.

            `backprop`: Runs `Graph.instrumented_backward` and serializes the
                backpropagation values (ie. saliency maps) of the requested
                `target_ids` as JSON to be viewable by the front-end visualizer

        Args:
            port: The port to run this server on.
            url_base: The base_url for the api endpoints. "/api/v1" by default.
        """

        server = Process(target=start_graph_server, args=(
            self, port, url_base))
        server.start()
        atexit.register(lambda server: server.terminate(), server)

        # sleep for 3 seconds to give the server some time to start up.
        # probably a better way to do this, but sanic's app.run is blocking
        # and doesn't support callbacks. SAD!
        time.sleep(1)

        if jupyter_widget:
            IPython.display.display(
                IPython.display.HTML(
                    "<iframe src='http://0.0.0.0:7060' width='800px' scrolling='no' height='500px' style='background-color:white; overflow: hidden;'></iframe>"))
        else:
            webbrowser.open("0.0.0.0:" + port)


def start_graph_server(graph, port=7060, url_base="/api/v1"):
    app = Sanic(log_config=None)

    CORS(app)

    @app.route("/")
    async def index(request):
        return await response.file(
            os.path.join(
                os.path.dirname(__file__),
                "viewer.html"))

    @app.route(url_base + "/graph_spec")
    async def graph_spec(request):
        return response.text(graph.serialize())

    @app.route(url_base + "/instrumented_forward")
    async def activations(request):
        forward = graph.instrumented_forward()
        for k in forward['activations']:
            forward['activations'][k]['inputs'] = list(
                forward['activations'][k]['inputs'])
            for i, inpt in enumerate(forward['activations'][k]['inputs']):
                forward['activations'][k]['inputs'][i] = inpt.tolist()
            forward['activations'][k]['output'] = forward['activations'][k]['output'].tolist()

        forward['result'] = forward['result'].tolist()

        return response.json(forward)

    app.run(host="0.0.0.0", port=port, log_config=None)


def alphabetical_ids(prev=''):
    """
    An infinite iterator over alphabetical id strings from 'a' to 'aa' to 'zz'.
    etc...
    """
    for char in ascii_lowercase:
        if str_lessthan(str(prev), str(char)):
            yield char

    for rest in alphabetical_ids():
        for char in ascii_lowercase:
            res = rest + char
            if str_lessthan(str(prev), str(res)):
                yield res


def str_lessthan(str1, str2):
    """
    Compares alphabetical IDs.
    """
    str1_val = 0
    str2_val = 0

    for i, str_char in enumerate(reversed(str1)):
        str1_val += (ord(str_char) - ord('a') + 1) * pow(26, i)

    for i, str_char in enumerate(reversed(str2)):
        str2_val += (ord(str_char) - ord('a') + 1) * pow(26, i)

    return str1_val < str2_val
