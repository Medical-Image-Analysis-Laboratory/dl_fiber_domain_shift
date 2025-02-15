import torch
import torch.nn as nn
from torchview import draw_graph

import monai
from monai.networks.blocks import Convolution, ResidualUnit
from typing import Sequence


class Counter:
    def __init__(self):
        self.counter = 0

    def __call__(self):
        self.counter += 1
        return self.counter - 1


class ConcatBlock(nn.Sequential):
    """
    A ConcatBlock is a sequence of layers where the initial input is concatenated to the output of all sequential layers, as the final output of the block.
    This has the effect of doubling the number of channels in the output of the block.

    Args:
        layers: sequence of nn.Module objects to define the individual layers of the concat block.
    """

    def __init__(self, layers: Sequence[nn.Module]):
        super().__init__()
        for i, l in enumerate(layers):
            self.add_module(f"layers{i}", l)

    def forward(self, x):
        x_0 = x
        for l in self.children():
            x = l(x)
        return torch.cat([x_0, x], dim=1)


class AddBlock(nn.Sequential):
    """
    An AddBlock is a sequence of layers where the initial input is added to the output of all sequential layers, as the final output of the block.
    This has the effect of keeping the number of channels in the output of the block the same as the input.

    Args:
        layers: sequence of nn.Module objects to define the individual layers of the add block.
    """

    def __init__(self, layers: Sequence[nn.Module]):
        super().__init__()
        for i, l in enumerate(layers):
            self.add_module(f"layers{i}", l)

    def forward(self, x):
        x_0 = x
        for l in self.children():
            x = l(x)
        return torch.add(x_0, x)


class DavoodNet(nn.Module):
    def __init__(
        self,
        spatial_dims,
        kernel_size,
        depth,
        n_feat_0,
        num_channel,
        num_class,
        dropout,
        act,
        norm,
    ):
        super().__init__()
        self.spatial_dims = spatial_dims
        self.kernel_size = kernel_size
        self.depth = depth
        self.n_feat_0 = n_feat_0
        self.num_channel = num_channel
        self.num_class = num_class
        self.dropout = dropout
        self.act = act
        self.norm = norm

        # Define the encoding path
        self.enc_blocks = nn.ModuleList()
        for level in range(depth):
            strd = 1 if level == 0 else 2
            block = self._conv_block(num_channel, n_feat_0, strd)
            self.enc_blocks.append(block)

            if level != 0:
                for i in range(1, level):
                    block = self._conv_block(n_feat_0, n_feat_0, strd)
                    self.enc_blocks.append(block)

                for level_reg in range(level):
                    level_diff = level - level_reg
                    n_feat = n_feat_0 * 2**level_reg

                    for j in range(level_diff):
                        block = self._conv_block(n_feat, n_feat, strd)
                        self.enc_blocks.append(block)

            n_feat = n_feat_0 * 2**level
            block = self._residual_block(n_feat, n_feat)
            self.enc_blocks.append(block)
            block = self._residual_block(n_feat, n_feat)
            self.enc_blocks.append(block)

        # Define the decoding path
        self.dec_blocks = nn.ModuleList()
        for level in range(depth - 2, -1, -1):
            block = self._conv_block(n_feat, n_feat // 2, 2, True)
            self.dec_blocks.append(block)

            n_concat = n_feat if level == depth - 2 else n_feat * 3 // 4
            n_feat = n_feat // 2 if level < depth - 2 else n_feat

            block = self._conv_block(n_concat, n_feat, 1)
            self.dec_blocks.append(block)

            block = self._residual_block(n_feat, n_feat)
            self.dec_blocks.append(block)
            block = self._residual_block(n_feat, n_feat)
            self.dec_blocks.append(block)

        # Define the output layer
        self.output_layer = self._conv_block(n_feat, num_class, 1, conv_only=True)

    def forward(self, x):
        feat_fine = [None] * (self.depth - 1)
        x0 = x

        enc_counter = Counter()
        # Encoding path
        for level in range(self.depth):
            x = self.enc_blocks[enc_counter()](x0)

            if level != 0:
                for i in range(1, level):
                    x = self.enc_blocks[enc_counter()](x)

                for level_reg in range(level):
                    x_0 = feat_fine[level_reg]
                    level_diff = level - level_reg
                    for j in range(level_diff):
                        x_0 = self.enc_blocks[enc_counter()](x_0)

                    x = torch.cat([x, x_0], dim=1)

            x_0 = x
            x = self.enc_blocks[enc_counter()](x)
            x = self.enc_blocks[enc_counter()](x)
            x += x_0

            if level < self.depth - 1:
                feat_fine[level] = x

        # Decoding path
        dec_counter = Counter()
        for level in range(self.depth - 2, -1, -1):
            x = self.dec_blocks[dec_counter()](x)
            x = torch.cat([feat_fine[level], x], dim=1)
            x = self.dec_blocks[dec_counter()](x)

            x_0 = x
            x = self.dec_blocks[dec_counter()](x)
            x = self.dec_blocks[dec_counter()](x)
            x += x_0

        # Output
        output = self.output_layer(x)

        return output

    def _conv_block(
        self, in_channels, out_channels, strides=1, is_transposed=False, conv_only=False
    ):
        return Convolution(
            spatial_dims=self.spatial_dims,
            in_channels=in_channels,
            out_channels=out_channels,
            strides=strides,
            kernel_size=self.kernel_size,
            act=self.act,
            norm=self.norm,
            dropout=self.dropout,
            conv_only=conv_only,
            is_transposed=is_transposed,
            adn_ordering="AD",
        )

    def _residual_block(
        self, in_channels, out_channels, strides=1, subunits=2, last_conv_only=False
    ):
        return ResidualUnit(
            spatial_dims=self.spatial_dims,
            in_channels=in_channels,
            out_channels=out_channels,
            strides=strides,
            kernel_size=self.kernel_size,
            subunits=subunits,
            act=self.act,
            norm=self.norm,
            dropout=self.dropout,
            last_conv_only=last_conv_only,
            adn_ordering="AD",
        )


if __name__ == "__main__":
    model = DavoodNet(
        spatial_dims=3,
        kernel_size=3,
        depth=3,
        n_feat_0=36,
        num_channel=6,
        num_class=45,
        dropout=0.1,
        act="relu",
        norm=None,
    )

    x = torch.randn(2, 6, 16, 16, 16)
    y = model(x)
    print(y.shape)

    graph = draw_graph(model, x).visual_graph

    # show the graph
    graph.view()

    # get the model size in MB
    print(
        f"Model size: {sum(p.numel() for p in model.parameters()) * 4 / 1024 / 1024:.2f}MB"
    )

    # save the model
    torch.save(model.state_dict(), "test/model.pt")
