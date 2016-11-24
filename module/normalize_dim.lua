require 'nn'
require 'cudnn'

do

    local Normalize, parent = torch.class('nn.normalize_dim', 'nn.Normalize')
    
    -- override the constructor to have the additional range of initialization
    function Normalize:__init(p,dim)
        parent.__init(self, )
                
        self:reset(mean,std)
    end
    
    -- override the :reset method to use custom weight initialization.        
    function SpatialConvolution:reset(mean,stdv)
        
        if mean and stdv then
            self.weight:normal(mean,stdv)
            self.bias:zero()
        else
            self.weight:normal(0,1)
            self.bias:zero()
        end
    end

end
