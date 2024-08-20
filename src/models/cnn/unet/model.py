import torch
import torch.nn as nn
from activation_fuctions.function_resolution import get_activation


class UnetModel(nn.Module):
    def __init__(
        self,
        in_channels: int,
        nb_classes: int,
        activation_fn_name: str,
        num_encoder_decoder_blocks: int,
        use_batchnorm: bool,
    ):
        super().__init__()
        self.encoder = UnetEncoder(
            in_channels=in_channels,
            activation_fn_name=activation_fn_name,
            num_blocks=num_encoder_decoder_blocks,
            use_batchnorm=use_batchnorm,
        )
        self.decoder = UnetDecoder(
            in_channels=self.encoder.out_channels,
            activation_fn_name=activation_fn_name,
            num_blocks=num_encoder_decoder_blocks,
            use_batchnorm=use_batchnorm,
            nb_classes=nb_classes,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip_connections, x = self.encoder(x)
        x = self.decoder(x, skip_connections)
        return x


class UnetEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        activation_fn_name: str,
        num_blocks: int,
        use_batchnorm: bool,
    ):
        super().__init__()
        self.blocks, last_block_out_channels = self.create_blocks(
            in_channels, activation_fn_name, num_blocks, use_batchnorm
        )
        self.out_channels = 1024
        self.output_layer = ConvBlock(
            in_channels=last_block_out_channels,
            out_channels=self.out_channels,
            kernel_size=3,
            stride=1,
            nb_convs=2,
            activation_fn_name=activation_fn_name,
            use_batchnorm=use_batchnorm,
        )

    def create_blocks(
        self,
        in_channels: int,
        activation_fn_name: str,
        num_blocks: int,
        use_batchnorm: bool,
    ):
        blocks = []
        out_channels = 64
        for _ in range(num_blocks):
            block = UnetEncoderBlock(
                in_channels, out_channels, activation_fn_name, use_batchnorm
            )
            blocks.append(block)
            in_channels = out_channels
            out_channels *= 2

        return nn.ModuleList(blocks), int(in_channels)

    def forward(self, x: torch.Tensor) -> tuple:
        skip_connections = []
        for block in self.blocks:
            conv_output, x = block(x)
            skip_connections.append(conv_output)

        x = self.output_layer(x)

        return skip_connections, x


class UnetEncoderBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        activation_fn_name: str,
        use_batchnorm: bool,
    ):
        super().__init__()
        self.conv_block = ConvBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            nb_convs=2,
            activation_fn_name=activation_fn_name,
            use_batchnorm=use_batchnorm,
        )

    def forward(self, x: torch.Tensor) -> tuple:
        conv_output = self.conv_block(x)
        pooled_output = torch.max_pool2d(conv_output, kernel_size=2, stride=2)
        return conv_output, pooled_output


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        nb_convs: int,
        activation_fn_name: str,
        use_batchnorm: bool,
    ):
        super().__init__()
        self.layers = self.create_layers(
            in_channels, out_channels, kernel_size, stride, nb_convs, use_batchnorm
        )
        self.out_channels = out_channels
        self.activation_fn = get_activation(activation_fn_name)

    def create_layers(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        nb_convs: int,
        use_batchnorm: bool,
    ):
        layers = []
        for _ in range(nb_convs):
            layers.append(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding="same",
                )
            )
            if use_batchnorm:
                layers.append(nn.BatchNorm2d(num_features=out_channels))
            in_channels = out_channels

        return nn.ModuleList(layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
            x = self.activation_fn(x)

        return x


class UnetDecoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        activation_fn_name: str,
        num_blocks: int,
        use_batchnorm: bool,
        nb_classes: int,
    ):
        super().__init__()
        self.blocks, last_block_out_channels = self.create_blocks(
            in_channels, activation_fn_name, num_blocks, use_batchnorm
        )
        self.output_layer = nn.Conv2d(
            in_channels=last_block_out_channels,
            out_channels=nb_classes,
            kernel_size=1,
            stride=1,
            padding="same",
        )

    def create_blocks(
        self,
        in_channels: int,
        activation_fn_name: str,
        num_blocks: int,
        use_batchnorm: bool,
    ):
        blocks = []
        out_channels = 512
        for _ in range(num_blocks):
            block = UnetDecoderBlock(
                int(in_channels), int(out_channels), activation_fn_name, use_batchnorm
            )
            blocks.append(block)
            in_channels = out_channels
            out_channels /= 2

        return nn.ModuleList(blocks), int(in_channels)

    def forward(self, x: torch.Tensor, skip_connections: list) -> torch.Tensor:
        for i, block in enumerate(self.blocks):
            skip_connection = skip_connections[-(i + 1)]
            x = block(x, skip_connection)

        x = self.output_layer(x)

        return x


class UnetDecoderBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        activation_fn_name: str,
        use_batchnorm: bool,
    ):
        super().__init__()
        self.up_conv = nn.ConvTranspose2d(
            in_channels=in_channels, out_channels=out_channels, kernel_size=2, stride=2
        )
        in_channels_after_concat = out_channels * 2
        self.conv_block = ConvBlock(
            in_channels=in_channels_after_concat,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            nb_convs=2,
            activation_fn_name=activation_fn_name,
            use_batchnorm=use_batchnorm,
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up_conv(x)
        x = torch.concat([x, skip], dim=1)
        x = self.conv_block(x)
        return x
