require 'GramMatrix'

local StyleLoss, parent = torch.class('nn.StyleLoss', 'nn.Module')

function StyleLoss:__init(strength)
    parent.__init(self)
    self.strength = strength
    self.target = torch.Tensor()
    self.loss = 0
    self.crit = nn.MSECriterion()
    self.mode = 'none'

    self.gram = nn.GramMatrix()
    self.G = nil
end

function StyleLoss:updateOutput(input)
    self.G = self.gram:forward(input)
    self.G:div(input:nElement())
    if self.mode == 'capture' then
        self.target:resizeAs(self.G):copy(self.G)
    elseif self.mode == 'loss' then
        self.loss = self.strength * self.crit:forward(self.G, self.target)
    end
    self.output = input
    return self.output
end

function StyleLoss:updateGradInput(input, gradOutput)
    if self.mode == 'loss' then
        local dG = self.crit:backward(self.G, self.target)
        dG:div(input:nElement())
        self.gradInput = self.gram:backward(input, dG)
        self.gradInput:mul(self.strength)
        self.gradInput:add(gradOutput)
    else
        self.gradInput = gradOutput
    end
    return self.gradInput
end