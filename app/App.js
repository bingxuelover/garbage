import React, { PureComponent } from "react"
import { file2Img, img2x } from "./utils";
import { Button, Progress } from 'antd'
import 'antd/dist/antd.css'
import * as tf from '@tensorflow/tfjs'
import intro from './intro'

const DATA_URL = "http://127.0.0.1:5000/"

class App extends PureComponent {
    state = {}

    async componentDidMount() {
        this.model = await tf.loadLayersModel(DATA_URL + '/model.json');
        this.model.summary()

        this.CLASSES = await fetch(DATA_URL + "/classes.json").then(res => res.json())
    }
    predict = async (file) => {
        const img = await file2Img(file);
        this.setState({
            imgSrc: img.src
        })
        setTimeout(() => {
            const pred = tf.tidy(() => {
                const x = img2x(img);
                return this.model.predict(x);
            })
            // pred.print()
            const results = pred.arraySync()[0]
                .map((score, i) => ({ score, label: this.CLASSES[i] }))
                .sort((a, b) => b.score - a.score);
            this.setState({
                results,
            })
        }, 0)
    }
    renderResult = (item) => {
        const finalScore = Math.round(item.score * 100);
        return (
            <tr key={item.label}>
                <td style={{ width: 80, padding: '5px 0' }}>{item.label}</td>
                <td><Progress percent={finalScore} status={finalScore === 100 ? 'success' : 'normal'} /></td>
            </tr>
        )
    }

    render() {
        const { imgSrc, results } = this.state
        const finalItem = results && { ...results[0], ...intro[results[0].label] }
        return (
            <div style={{ padding: 20, maxWidth: 750 }}>
                <Button
                    type="primary"
                    size="large"
                    onClick={() => this.upload.click()}
                >选择图片识别</Button>
                <input
                    type="file"
                    onChange={e => this.predict(e.target.files[0])}
                    ref={el => { this.upload = el }}
                    style={{ display: "none" }}
                />
                <div>{this.className}</div>
                {imgSrc && <div style={{ marginTop: 20, textAlign: "center" }}>
                    <img src={imgSrc} style={{ maxWidth: '100%', height: 200 }} />
                </div>}
                {finalItem && <div style={{ display: "flex", alignItems: "flex-start", marginTop: 20 }}>
                    <img src={finalItem.icon} width={120} />
                    <div>
                        <h2 style={{ color: finalItem.color }}>{finalItem.label}</h2>
                        <p style={{ color: finalItem.color }}>{finalItem.intro}</p>
                    </div>
                </div>}
                {results && <div style={{ margnTop: 20 }}>
                    <table style={{ width: '100%' }}>
                        <tbody>
                            <tr>
                                <td>类别</td>
                                <td>匹配度</td>
                            </tr>
                            {results.map(this.renderResult)}
                        </tbody>
                    </table>
                </div>}
            </div>
        )
    }
}

export default App