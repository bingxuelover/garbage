const tf = require("@tensorflow/tfjs-node")
const getData = require("./data");

const TRAIN_DIR = 'image'
const OUTPUT_DIR = 'output'
const MOBILENET_URL = 'http://ai-sample.oss-cn-hangzhou.aliyuncs.com/pipcook/models/mobilenet/web_model/model.json'

const main = async () => {
    /**
     * 加载数据
     */
    const { ds, classes } = await getData(TRAIN_DIR, OUTPUT_DIR);
    // console.log(xs,ys, classes);

    /**
     * 定义模型
     */
    const mobilenet = await tf.loadLayersModel(MOBILENET_URL);
    mobilenet.summary()
    // console.log(mobilenet.layers.map((l, i) => [l.name, i]));
    //截取模型
    const model = tf.sequential();
    for (let i = 0; i <= 86; i++) {
        const layer = mobilenet.layers[i];
        layer.trainable = false;
        model.add(layer);
    }
    //高维数据摊平
    model.add(tf.layers.flatten());
    //隐藏层
    model.add(tf.layers.dense({
        units: 10,
        activation: 'relu'//激活函数，处理非线性函数
    }));
    //输出层
    model.add(tf.layers.dense({
        units: classes.length,
        activation: 'softmax'//解决多分类问题
    }))

    /**
     * 训练模型
     */
    //定义损失函数和优化器
    model.compile({
        loss: "sparseCategoricalCrossentropy",
        optimizer: tf.train.adam(),
        metrics: ['acc']
    });
    //训练20遍
    await model.fitDataset(ds, { epochs: 20 });
    //保存
    await model.save(`file://${process.cwd()}/${OUTPUT_DIR}`);
}

main()