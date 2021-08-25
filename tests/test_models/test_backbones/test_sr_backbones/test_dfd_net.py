import pytest
import torch
import torch.nn as nn

from mmedit.models import build_backbone


def test_dfd_net():
    
    model_cfg = dict(
        type='DFDNet',
        mid_channels=64,
        dictionary_path='/mnt/lustre/liyinshuo/01-git-clone/DFDNet/DictionaryCenter512')

    # build model
    model = build_backbone(model_cfg)

    # test attributes
    assert model.__class__.__name__ == 'DFDNet'

    # prepare data
    inputs = torch.rand(1, 3, 512, 512)
    part_locations = [torch.tensor([[146, 184, 225, 263]]), 
                      torch.tensor([[283, 179, 374, 270]]), 
                      torch.tensor([[229, 296, 282, 349]]), 
                      torch.tensor([[195, 305, 323, 433]])]
    [torch.rand(1, 4) for _ in range(4)]
    targets = torch.rand(1, 3, 512, 512)

    # prepare loss
    loss_function = nn.L1Loss()

    # prepare optimizer
    optimizer = torch.optim.Adam(model.parameters())

    # test on cpu
    output = model(inputs, part_locations)
    optimizer.zero_grad()
    loss = loss_function(output, targets)
    loss.backward()
    optimizer.step()
    assert torch.is_tensor(output)
    assert output.shape == targets.shape

    # test on gpu
    if torch.cuda.is_available():
        model = model.cuda()
        optimizer = torch.optim.Adam(model.parameters())
        inputs = inputs.cuda()
        part_locations = [p.cuda() for p in part_locations]
        targets = targets.cuda()
        output = model(inputs, part_locations)
        optimizer.zero_grad()
        loss = loss_function(output, targets)
        loss.backward()
        optimizer.step()
        assert torch.is_tensor(output)
        assert output.shape == targets.shape

    # with pytest.raises(OSError):
    #     model.init_weights('')
    # with pytest.raises(TypeError):
    #     model.init_weights(1)


if __name__ == '__main__':
    test_dfd_net()
