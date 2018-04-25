-- Rewritten based on https://github.com/jcjohnson/neural-style
-- Not normalizing model and gradients
require 'nn'
require 'optim'
require 'loadcaffe'
require 'image'
require 'vgg-img-process'
require 'ContentLoss'
require 'StyleLoss'


local cmd = torch.CmdLine()

-- Image options
cmd:option('-style_image', 'examples/inputs/seated-nude.jpg', 'Style target image')
cmd:option('-content_image', 'examples/inputs/tubingen.jpg', 'Content target image')
cmd:option('-image_size', 512, 'Maximum height / width of generated image')

-- Optimization options
cmd:option('-content_weight', 5e0)
cmd:option('-style_weight', 1e2)
cmd:option('-learning_rate', 1e1)
cmd:option('-num_iterations', 1000)

-- GPU options
cmd:option('-gpu', '0', 'Zero-indexed ID of the GPU to use; for CPU mode set -gpu = -1')

-- Model options
cmd:option('-proto_file', '/home/socurites/git/torch/neural-style/models/VGG_ILSVRC_19_layers_deploy.prototxt')
cmd:option('-model_file', '/home/socurites/git/torch/neural-style/models/VGG_ILSVRC_19_layers.caffemodel')
cmd:option('-backend', 'nn', 'nn|cudnn|clnn')
cmd:option('-pooling', 'avg', 'max|avg')

-- Layer options
cmd:option('-content_layers', 'relu4_2', 'layers for content')
cmd:option('-style_layers', 'relu1_1,relu2_1,relu3_1,relu4_1,relu5_1', 'layers for style')

-- Output options
cmd:option('-print_iter', 50)
cmd:option('-save_iter', 100)

function main(params)
    dtype = 'torch.FloatTensor'

    -- Load pre-traing vgg19 model
    cnn = loadcaffe.load(params.proto_file, params.model_file, params.backend):type(dtype)

    -- Load content image
    content_image = image.load(params.content_image, 3)
    content_image = image.scale(content_image, params.image_size, 'bilinear')

    content_image_caffe = vgg_img_process().preprocess(content_image)

    -- Load style image
    style_image = image.load(params.style_image, 3)
    style_image = image.scale(style_image, params.image_size, 'bilinear')
    style_image_caffe = vgg_img_process().preprocess(style_image)

    -- Set layers
    content_layers = params.content_layers:split(",")
    style_layers = params.style_layers:split(",")

    -- Define model
    content_losses, style_losses = {}, {}
    next_content_idx, next_style_idx = 1, 1

    net = nn.Sequential()

    for i = 1, #cnn do
         if next_content_idx <= #content_layers or next_style_idx <= #style_layers then
            local layer = cnn:get(i)
            local name = layer.name
            local layer_type = torch.type(layer)
            local is_pooling = (layer_type == 'cudnn.SpatialMaxPooling' or layer_type == 'nn.SpatialMaxPooling')
            if is_pooling and params.pooling == 'avg' then
                assert(layer.padW == 0 and layer.padH == 0)
                local kW, kH = layer.kW, layer.kH
                local dW, dH = layer.dW, layer.dH
                local avg_pool_layer = nn.SpatialAveragePooling(kW, kH, dW, dH):type(dtype)
                local msg = 'Replacing max pooling at layer %d with average pooling'
                print(string.format(msg, i))
                net:add(avg_pool_layer)
            else
                net:add(layer)
            end
            if name == content_layers[next_content_idx] then
                print("Setting up content layer", i, ":", layer.name)
                local loss_module = nn.ContentLoss(params.content_weight):type(dtype)
                net:add(loss_module)
                table.insert(content_losses, loss_module)
                next_content_idx = next_content_idx + 1
            end
            if name == style_layers[next_style_idx] then
                print("Setting up style layer  ", i, ":", layer.name)
                local loss_module = nn.StyleLoss(params.style_weight):type(dtype)
                net:add(loss_module)
                table.insert(style_losses, loss_module)
                next_style_idx = next_style_idx + 1
            end
        end
    end
    net:type(dtype)

    -- We don't need the base CNN anymore, so clean it up to save memory.
    cnn = nil
    collectgarbage()

    -- Capture content targets
    for i = 1, #content_losses do
        content_losses[i].mode = 'capture'
    end
    for j = 1, #style_losses do
        style_losses[j].mode = 'none'
    end

    print 'Capturing content targets'
    net:forward(content_image_caffe:type(dtype))


    -- Capture style targets
    for i = 1, #content_losses do
        content_losses[i].mode = 'none'
    end
    for j = 1, #style_losses do
        style_losses[j].mode = 'capture'
    end
    print 'Capturing style targets'
    net:forward(style_image_caffe:type(dtype))


    -- Set all loss modules to loss mode
    for i = 1, #content_losses do
        content_losses[i].mode = 'loss'
    end
    for i = 1, #style_losses do
        style_losses[i].mode = 'loss'
    end


    -- Initialize the image
    img = torch.randn(content_image:size()):float():mul(0.001)
    img = img:type(dtype)


    -- Set optimzer
    -- Use adam instaed of lbfgs
    optim_state = {
        learningRate = params.learning_rate,
    }


    -- Run it through the network once to get the proper size for the gradient
    -- All the gradients will come from the extra loss modules, so we just pass
    -- zeros into the top of the net on the backward pass.
    local y = net:forward(img)
    local dy = img.new(#y):zero()

    -- Define feval
    num_calls = 0
    function feval(x)
        num_calls = num_calls + 1
        net:forward(x)
        local grad = net:updateGradInput(x, dy)
        local loss = 0
        for _, mod in ipairs(content_losses) do
            loss = loss + mod.loss
        end
        for _, mod in ipairs(style_losses) do
            loss = loss + mod.loss
        end
        print_losses(params.num_iterations, params.print_iter, num_calls, loss, content_losses, style_losses)
        save_output_img(params.save_iter, params.num_iterations, num_calls, img)

        collectgarbage()
        return loss, grad:view(grad:nElement())
    end

    print('Running optimization with ADAM')
    for t = 1, params.num_iterations do
        local x, losses = optim.adam(feval, img, optim_state)
    end
end


function print_losses(num_iterations, print_iter, t, loss, content_losses, style_losses)
    if (print_iter > 0 and t % print_iter == 0) then
        print(string.format('Iteration %d / %d', t, num_iterations))
        for i, loss_module in ipairs(content_losses) do
            print(string.format('  Content %d loss: %f', i, loss_module.loss))
        end
        for i, loss_module in ipairs(style_losses) do
            print(string.format('  Style %d loss: %f', i, loss_module.loss))
        end
        print(string.format('  Total loss: %f', loss))
    end
end

function save_output_img(save_iter, num_iterations, t, img)
    if (save_iter > 0 and t % save_iter == 0) or t == num_iterations then
        local disp = vgg_img_process().deprocess(img:double())

        disp = image.minmax{tensor=disp, min=0, max=1}
        filename = string.format('$s_%d.%s', 'output', iteration, 'png')

        image.save(filename, disp)
    end
end

local params = cmd:parse(arg)
main(params)