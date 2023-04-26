import astroid
import logging
import inspect

from ._base_node import NodeEntityBase
from ._data_class_node import DataClassNode
from ._class_node import ClassNode
from ._function_node import FunctionNode
from ._variable_node import VariableNode
from apistub import Navigation, Kind, NavigationTag

filter_globals = lambda x: isinstance(x, VariableNode)
filter_function = lambda x: isinstance(x, FunctionNode)
filter_class = lambda x: isinstance(x, ClassNode)


class ModuleNode(NodeEntityBase):
    """ModuleNode represents module level node and all it's children
    :param str: namespace
    :param module: module
    :param dict: node_index
    """

    def __init__(self, namespace, module, node_index, pkg_root_namespace):
        super().__init__(namespace, None, module)
        self.namespace_id = self.generate_id()
        self.node_index = node_index
        self.pkg_root_namespace = pkg_root_namespace
        self._inspect()

    def _inspect(self):
        """Imports module, identify public entities in module and inspect them recursively
        """
        # Parse public entities only if __all is present. Otherwise all Classes and Functions not starting with "_" can be included.
        public_entities = []
        if hasattr(self.obj, "__all__"):
            public_entities = getattr(self.obj, "__all__")

        openai_globals = []
        if self.obj.__name__ == "openai":
            openai_globals = ["api_key", "api_key_path", "organization", "api_base", "api_type", "api_version", "verify_ssl_certs", "proxy",
                            "app_info", "enable_telemetry", "ca_bundle_path", "debug", "log", "aiosession"]
            for g in openai_globals:
                if g not in public_entities:
                    public_entities.append(g)

        # find class and function nodes in module
        for name, member_obj in inspect.getmembers(self.obj):
            if self._should_skip_parsing(name, member_obj, public_entities):
                continue

            if inspect.isclass(member_obj):
                class_type = ClassNode
                try:
                    # see if a class is annotated as a dataclass
                    node = astroid.extract_node(inspect.getsource(member_obj))
                    if node.decorators:
                        for item in node.decorators.nodes:
                            if getattr(item, "name", None) == "dataclass":
                                class_type = DataClassNode
                                break
                except:
                    pass
                class_node = class_type(
                    name=name,
                    namespace=self.namespace,
                    parent_node=self,
                    obj=member_obj,
                    pkg_root_namespace=self.pkg_root_namespace
                )
                key = "{0}.{1}".format(self.namespace, class_node.name)
                self.node_index.add(key, class_node)
                self.child_nodes.append(class_node)
            elif inspect.isroutine(member_obj):
                func_node = FunctionNode(self.namespace, self, obj=member_obj, is_module_level=True)
                key = "{0}.{1}".format(self.namespace, func_node.name)
                self.node_index.add(key, func_node)
                self.child_nodes.append(func_node)
            elif name in openai_globals:
                variable_node = VariableNode(namespace=self.namespace, parent_node=self, name=name, type_name=None, value=member_obj, is_ivar=False)
                print(f"adding variable node: {variable_node}: name: {name} obj={member_obj}")
                key = "{0}.{1}".format(self.namespace, variable_node.name)
                self.node_index.add(key, variable_node)
                self.child_nodes.append(variable_node)
            else:
                logging.debug("Skipping unknown type member in module: {}".format(name))

    def _should_skip_parsing(self, name, member_obj, public_entities):
        # If module has list of published entities ( __all__) then include only those members
        if public_entities and name not in public_entities:
            logging.debug(
                "Object is not listed in __all__. Skipping object {}".format(name)
            )
            return True

        # Skip any private members
        if name.startswith("_"):
            logging.debug("Skipping object {}".format(name))
            return True

        # Skip any member in module level that is defined in external or built in package
        if hasattr(member_obj, "__module__"):
            return not getattr(member_obj, "__module__").startswith(self.pkg_root_namespace)
        # Don't skip member if module name is not available. This is just to be on safer side
        return False

    def generate_tokens(self, apiview):
        """Generates token for the node and it's children recursively and add it to apiview
        :param ApiView: apiview
        """
        if self.child_nodes:
            # write globals
            for g in filter(filter_globals, self.child_nodes):
                g.generate_tokens(apiview)
                apiview.set_blank_lines(1)

            # Add name space level functions first
            for c in filter(filter_function, self.child_nodes):
                c.generate_tokens(apiview)
                apiview.set_blank_lines(2)

            # Add classes
            for c in filter(filter_class, self.child_nodes):
                c.generate_tokens(apiview)
                apiview.set_blank_lines(2)

    def get_navigation(self):
        """Generate navigation tree recursively by generating Navigation obejct for classes and functions in name space
        """
        if self.child_nodes:
            navigation = Navigation(self.namespace_id, self.namespace_id)
            navigation.tags = NavigationTag(Kind.type_module)
            # Generate child navigation for each child nodes
            for c in filter(filter_function, self.child_nodes):
                child_nav = Navigation(c.name, c.namespace_id)
                child_nav.tags = NavigationTag(Kind.type_method)
                navigation.add_child(child_nav)

            for c in filter(filter_class, self.child_nodes):
                child_nav = Navigation(c.name, c.namespace_id)
                child_nav.tags = NavigationTag(Kind.type_enum if c.is_enum else Kind.type_class)
                navigation.add_child(child_nav)
            return navigation
