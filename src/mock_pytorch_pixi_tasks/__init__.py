from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("mock_pixi_tasks")
except PackageNotFoundError:
    __version__ = "uninstalled"