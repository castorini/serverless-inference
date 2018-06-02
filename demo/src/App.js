import $ from "jquery";
import React, { Component } from 'react';
import { Alert, Button, Container, InputGroup, Input, Jumbotron } from 'reactstrap';
import './App.css';
import kimCNNData from './kim-cnn-data.json';

class App extends Component {
  constructor(props) {
    super(props);
    this.state = {
      sm_cnn: {
        score: ' '
      },
      kim_cnn: {
        score: ' '
      }
    }

    this.evaluate_smcnn = this.evaluate_smcnn.bind(this);
    this.evaluate_kimcnn = this.evaluate_kimcnn.bind(this);
  }

  evaluate_smcnn() {
    $.ajax({
      type: 'POST',
      url: process.env.REACT_APP_SM_CNN_URL,
      data: JSON.stringify({
        sent1: $('#sm-cnn-sent-1').val(),
        sent2: $('#sm-cnn-sent-2').val()
      }),
      dataType: 'json',
      success: (data) => {
        var softmaxDenom = Math.exp(data.score[0][0]) + Math.exp(data.score[0][1]);
        var score = Math.exp(data.score[0][0]) / softmaxDenom;
        this.setState({
          sm_cnn: {
            score: score.toFixed(2)
          }
        });
      }
    });
  }

  evaluate_kimcnn() {
    console.log($('#kim-cnn-input').val());
    $.ajax({
      type: 'POST',
      url: process.env.REACT_APP_KIM_CNN_URL,
      data: JSON.stringify({
        input: $('#kim-cnn-input').val()
      }),
      dataType: 'json',
      success: (data) => {
        var softmaxDenom = 0;
        for (var i = 0; i < 6; i++) {
          softmaxDenom += Math.exp(data.output[i]);
        }
        var score = 0;
        for (i = 0; i < 6; i++) {
          score += i * Math.exp(data.output[i]) / softmaxDenom;
        }
        this.setState({
          kim_cnn: {
            score: score.toFixed(2)
          }
        });
      }
    });
  }

  render() {
    const kimCNNListItems = kimCNNData.map((d, i) => <option key={'k'+i}>{d.sentence}</option>);
    return (
      <div className="App">
        <Jumbotron>
          <Container>
            <h1 className="display-3">Serverless Inference Demo</h1>
            <p className="lead">Try out text ranking and sentence classification models deployed on AWS Lambda</p>
          </Container>
        </Jumbotron>
        <div className="pt-3 pb-5 mb-3 px-3">
          <Container>
            <div>
              <h2 className="display-4">Kim CNN</h2>
              <p className="lead">Sentence classification</p>
            </div>
            <InputGroup>
              <Input type="select" name="select" id="kim-cnn-input">
                {kimCNNListItems}
              </Input>
            </InputGroup>
            <br />
            <Alert color="success">
              {'Score: '}{this.state.kim_cnn.score}
            </Alert>
            <Button className="float-right"
              color="secondary"
              onClick={this.evaluate_kimcnn}>
              Evaluate
            </Button>
          </Container>
        </div>
        <div className="bg-light pt-3 pb-5 px-3">
          <Container>
            <div>
              <h2 className="display-4">SM-CNN</h2>
              <p className="lead">Short text pair ranking</p>
              <InputGroup>
                <Input id="sm-cnn-sent-1" placeholder="how are glacier caves formed" />
              </InputGroup>
              <br />
              <InputGroup>
                <Input id="sm-cnn-sent-2" placeholder="the ice facade is approximately 60 m high" />
              </InputGroup>
              <br />
              <Alert color="success">
                {'Score: '}{this.state.sm_cnn.score}
              </Alert>
              <Button className="float-right"
                color="secondary"
                onClick={this.evaluate_smcnn}>
                Evaluate
              </Button>
            </div>
          </Container>
        </div>
      </div>
    );
  }
}

export default App;
