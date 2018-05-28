import $ from "jquery";
import React, { Component } from 'react';
import { Alert, Button, Container, InputGroup, Input, Jumbotron } from 'reactstrap';
import './App.css';

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
        sent1: 'how are glacier caves formed',
        sent2: 'the ice facade is approximately 60 m high'
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
    $.ajax({
      type: 'POST',
      url: process.env.REACT_APP_KIM_CNN_URL,
      data: JSON.stringify({
        input: 'the movie has a happy ending'
      }),
      dataType: 'json',
      success: (data) => {
        var softmaxDenom = 0;
        for (var i = 0; i < 6; i++) {
          softmaxDenom += Math.exp(data.output[i]);
        }
        var score = 0;
        for (var i = 0; i < 6; i++) {
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
              <h2 className="display-4">SM-CNN</h2>
              <p className="lead">Short text pair ranking</p>
              <InputGroup>
                <Input placeholder="how are glacier caves formed" />
              </InputGroup>
              <br />
              <InputGroup>
                <Input placeholder="the ice facade is approximately 60 m high" />
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
        <div className="bg-light pt-3 pb-5 px-3">
          <Container>
            <div>
              <h2 className="display-4">Kim CNN</h2>
              <p className="lead">Sentence classification</p>
            </div>
            <InputGroup>
              <Input placeholder="the movie has a happy ending" />
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
      </div>
    );
  }
}

export default App;
