--[[
-- about gradInput:add(gradOutput): https://github.com/jcjohnson/neural-style/issues/362
 ]]
local ContentLoss, parent = torch.class('nn.ContentLoss', 'nn.Module')

function ContentLoss:__init(strength)
    parent.__init(self)
    self.strength = strength
    self.target = torch.Tensor()
    self.loss = 0
    self.crit = nn.MSECriterion()
    self.mode = 'none'
end

function ContentLoss:updateOutput(input)
    if self.mode == 'loss' then
        self.loss = self.crit:forward(input, self.target) * self.strength
    elseif self.mode == 'capture' then
        self.target:resizeAs(input):copy(input)
    end
    self.output = input
    return self.output
end

function ContentLoss:updateGradInput(input, gradOutput)
    if self.mode == 'loss' then
        if input:nElement() == self.target:nElement() then
            self.gradInput = self.crit:backward(input, self.target)
        end
        self.gradInput:mul(self.strength)
        self.gradInput:add(gradOutput)
    else
        self.gradInput:resizeAs(gradOutput):copy(gradOutput)
    end
    return self.gradInput
end