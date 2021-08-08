from mmcv.cnn import ConvModule
from mmcv.cnn.bricks.registry import PLUGIN_LAYERS



@PLUGIN_LAYERS.register_module()
class ConvModuleMixBN(ConvModule):
    
    def forward(self, x, activate=True, norm=True, domain=None):
        for layer in self.order:
            if layer == 'conv':
                if self.with_explicit_padding:
                    x = self.padding_layer(x)
                x = self.conv(x)
            elif layer == 'norm' and norm and self.with_norm:
                x = self.norm(x, domain=domain)
            elif layer == 'act' and activate and self.with_activation:
                x = self.activate(x)
        return x