require 'nn'
require 'cudnn'

do

    local CMul, parent = torch.class('nn.customCMul', 'nn.CMul')
    
    -- override the :reset method to use custom weight initialization.        
    function CMul:reset(init_value)
        
        if init_value then
            self.weight = torch.Tensor(self.size):fill(init_value)
        else
            self.weight:uniform(-1./math.sqrt(self.weight:nElement()),1./math.sqrt(self.weight:nElement()))
        end
    end

end
