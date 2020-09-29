import * as tf from '@tensorflow/tfjs'

/**
 * 上传数据转化为图片
 * @param {*} f 
 */
export function file2Img(f) {
    return new Promise(reslove => {
        const reader = new FileReader();
        reader.readAsDataURL(f);
        reader.onload = function (e) {
            const img = document.createElement('img');
            img.src = e.target.result;
            img.width = 224;
            img.height = 224;
            img.onload = () => reslove(img)
        }
    })
}

/**
 * 图片转化为tensor
 * @param {*} imgEl 
 */
export function img2x(imgEl) {
    return tf.tidy(() => {
        return tf.browser.fromPixels(imgEl)
            .toFloat().sub(255 / 2).div(255 / 2)
            .reshape([1, 224, 224, 3]);
    })
}