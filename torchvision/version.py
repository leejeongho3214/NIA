__version__ = '0.16.0+cu118'
git_version = 'a90e584667fc3a7d85485764245e0db92387aca1'
from torchvision.extension import _check_cuda_version
if _check_cuda_version() > 0:
    cuda = _check_cuda_version()
