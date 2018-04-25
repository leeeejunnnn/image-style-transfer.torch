--[[
    -- Preprocess an image before passing it to a Caffe model.
    -- We need to rescale from [0, 1] to [0, 255], convert from RGB to BGR,
    -- and subtract the mean pixel.
 ]]

function vgg_img_process()
    local util = {}

    function util.preprocess(img)
        img = util.rgb2bgr(img)
        img = util.scale_up(img)
        img = util.subtract_mean(img)

        return img:float()
    end

    function util.deprocess(img)
        img = util.add_mean(img)
        img = util.rgb2bgr(img)
        img = util.scale_down(img)

        return img
    end

    function util.rgb2bgr(img)
        local perm = torch.LongTensor{3, 2, 1}
        img = img:index(1, perm)

        return img
    end

    function util.scale_up(img)
        return img:mul(256.0)
    end

    function util.subtract_mean(img)
        local mean_pixel = torch.DoubleTensor({103.939, 116.779, 123.68})
        mean_pixel = mean_pixel:view(3, 1, 1):expandAs(img)
        img:add(-1, mean_pixel)

        return img
    end

    function util.scale_down(img)
        return img:div(256.0)
    end

    function util.add_mean(img)
        local mean_pixel = torch.DoubleTensor({103.939, 116.779, 123.68})
        mean_pixel = mean_pixel:view(3, 1, 1):expandAs(img)
        img = img + mean_pixel

        return img
    end

    return util
end