require 'nn'
require 'cudnn'

do

    local SpatialDilatedConvolution, parent = torch.class('nn.normalDilatedConv', 'nn.SpatialDilatedConvolution')
    
    -- override the constructor to have the additional range of initialization
    function SpatialDilatedConvolution:__init(nInputPlane, nOutputPlane, kW, kH, dW, dH, padW, padH, dilationW, dilationH, mean, std)
        parent.__init(self,nInputPlane, nOutputPlane, kW, kH, dW, dH, padW, padH, dilationW, dilationH)
                
        self:reset(mean,std)
    end
    
    -- override the :reset method to use custom weight initialization.        
    function SpatialDilatedConvolution:reset(mean,stdv)
        
        if mean and stdv then
            self.weight:normal(mean,stdv)
            self.bias:zero()
        else
            self.weight:normal(0,1)
            self.bias:zero()
        end
    end

end
