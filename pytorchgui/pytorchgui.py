import json
from datetime import datetime
from torch.autograd import Variable, Function
from torch.nn import Parameter
from string import ascii_lowercase
import inspect
import copy


class Graph:
    """
    A graph structure used to analyse a pytorch graph for pytorchgui
    """

    def __init__(self, module, inputs):
        self.module = module
        self.backward_graph = {}
        self.input_vars = []
        self.node_id_map = {}
        self.id_generator = alphabetical_ids()

        self._gen_backward_deps(module(inputs).creator)

    def _gen_backward_deps(self, node):
        """
        Recursively traverses the dependencies for a node and builds up the
        `Graph.backward_graph` mapping unique layers to their serializable IDs
        and their dependencies.

        If any layer is new (hasn't been seen by this PyTorchGraph object
        during a previous forward pass) then it will be added to both
        `node_id_map` and `backward_graph`. This is important as for dynamic
        neural networks, the layers actually being used and activated may not
        be included in the graph logic itself (if statements, etc...)

        Args:
            node : any pytorch.nn.autograd node.
                When being used non-recursively (outside this function),
                generally applied to the output variable of a module given
                inputs.

        Returns:
            The node_id of the node being passed in to the function.
            If the node doesn't already exist in the graph it will be assigned.
            This is mostly only useful for recursive purposes.
        """

        if node not in self.node_id_map:
            node_id = next(self.id_generator)
            self.node_id_map[node] = node_id

            self.backward_graph[node_id] = {
                'id': node_id,
                'type': type(node),
                'requires_grad': node.requires_grad,
                'obj': node
            }

            if isinstance(node, Parameter):
                self.backward_graph[node_id]['data'] = {
                    'type': type(node.data),
                    'size': node.data.size()
                }

            elif isinstance(node, Function):
                self.backward_graph[node_id]['parameters'] = []
                self.backward_graph[node_id]['dependencies'] = []

                for dep, _ in node.previous_functions:
                    dep_id = self._gen_backward_deps(dep)
                    if isinstance(dep, Parameter):
                        self.backward_graph[node_id]['parameters'] \
                            .append(self.backward_graph[dep_id])
                    else:
                        self.backward_graph[node_id]['dependencies'] \
                            .append(self.backward_graph[dep_id])

            elif isinstance(node, Variable):
                self.backward_graph[node_id]['data'] = {
                    'type': type(node.data),
                    'size': node.data.size()
                }

                if hasattr(node, 'previous_functions'):
                    self.backward_graph[node_id]['dependencies'] = []
                    for dep, i in node.previous_functions:
                        dep_id = self._gen_backward_deps(dep)
                        self.backward_graph[node_id]['dependencies'] \
                            .append(self.backward_graph[dep_id])
                else:
                    if not node.requires_grad:
                        self.input_vars.append(node_id)

            else:
                raise Exception(
                    "node type unaccounted for!: " + type(node).__name__)
        else:
            node_id = self.node_id_map[node]

        return node_id

    def serialize(self):
        """
        Serializes the structure of the backward graph into a JSON format.
        This JSON addresses each node by a hash, providing its type, an object
        of its parameters, and its dependant nodes.

        Returns:
            A Json serialized version of this graph
        """
        serializable_graph = {
            'name': self.module.__class__.__name__,
            'date': str(datetime.now()),
            'input_vars': self.input_vars,
            'ops': {},
            'parameters': {}
        }

        for node_id, node in self.backward_graph.items():
            dict_key = 'parameters' if node['type'] == Parameter else 'ops'

            serializable_graph[dict_key][node_id] = {}

            node_dict = {}

            node_dict['type'] = node['type'].__name__

            if 'data' in node:
                node_dict['data'] = {
                    'type': node['data']['type'].__name__,
                    'size': node['data']['size']
                }

            # replace recursive with normalized ID's
            if 'parameters' in node:
                node_dict['parameters'] = \
                    [param['id'] for param in node['parameters']]

            if 'dependencies' in node:
                node_dict['dependencies'] = \
                    [dep['id'] for dep in node['dependencies']]

            serializable_graph[dict_key][node_id] = node_dict

        return json.dumps(serializable_graph)

    def instrumented_forward(self, inputs, target_ids='all'):
        """
        Completes a forward pass throughout the entire graph for the given
        input, returning the dict of target_ids mapping to their activations
        in this forward pass.

        Args:
            inputs : tuple of input tensor variables
                Input tensor variable to be used in place of the calculated
                input node of the graph.

            target_ids : list of node_ids or 'all'
                List of node_ids corresponding to the nodes of which the
                instrumented activations are requested. IE passing the id for a
                linear layer will include the output activation of that linear
                layer in the response.

        Returns:
            dict of target_id => activations as a numpy array
        """

        # generate the backward dependencies off this particular input
        self._gen_backward_deps(self.module(inputs).creator)

        if target_ids == 'all':
            target_ids = [node_id for node_id in self.backward_graph]

        activations = {}

        def fill_activations(node_id):
            if node_id not in activations:
                node_dict = self.backward_graph[node_id]
                node = node_dict['obj']

                if isinstance(node, Variable) or isinstance(node, Parameter):
                    activations[node_id] = node

                elif isinstance(node, Function):
                    args = ()

                    if 'dependencies' in node_dict:
                        for dep in node_dict['dependencies']:
                            fill_activations(dep['id'])
                            args += (activations[dep['id']],)

                    if 'parameters' in node_dict:
                        for param in node_dict['parameters']:
                            fill_activations(param['id'])
                            args += (activations[param['id']],)

                    activations[node_id] = node.forward(*args)

            return activations[node_id]

        target_activations = {}

        for target_id in target_ids:
            fill_activations(target_id)
            target_activations[target_id] = activations[target_id].data.numpy()

        return target_activations

    def serve(self, port=7060, url_base="/api/v1"):
        """
        Serves the current graph using a sanic webserver, providing a few api
        end-points:

            `graph_spec`: Serves the result of `Graph.serialize`.

            `activations`: Runs `Graph.instrumented_forward` and serializes the
                activations of the requested `target_ids` (POST data attribute)
                as JSON to be viewable by the front-end visualizer.

            `backprop`: Runs `Graph.instrumented_backward` and serializes the
                backpropagation values (ie. saliency maps) of the requested
                `target_ids` as JSON to be viewable by the front-end visualizer.

        Args:
            port: The port to run this server on.
            url_base: The base_url for the api endpoints. "/api/v1" by default.
        """
        # TODO this
        pass

    def __repr__(self):
        return self.module.__repr__()


def alphabetical_ids():
    """
    An infinite iterator over alphabetical id strings from 'a' to 'aa' to 'zz'.
    etc...
    """
    for char in ascii_lowercase:
        yield char
    for char in ascii_lowercase:
        for rest in alphabetical_ids():
            yield char + rest
