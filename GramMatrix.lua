local GramMatrix, parent = torch.class('nn.GramMatrix', 'nn.Module')

function GramMatrix:__init()
    parent.__init(self)
end

function GramMatrix:updateOutput(input)
    assert(input:dim() == 3)
    local C, H, W = input:size(1), input:size(2), input:size(3)
    local x_flat = input:view(C, H * W)
    self.output:resize(C, C)
    self.output:mm(x_flat, x_flat:t())
    return self.output
end

function GramMatrix:updateGradInput(input, gradOutput)
    assert(input:dim() == 3 and input:size(1))
    local C, H, W = input:size(1), input:size(2), input:size(3)
    local x_flat = input:view(C, H * W)
    self.gradInput:resize(C, H * W):mm(gradOutput, x_flat)
    self.gradInput:addmm(gradOutput:t(), x_flat)
    self.gradInput = self.gradInput:view(C, H, W)
    return self.gradInput
end