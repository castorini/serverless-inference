import React, { Component } from 'react';
import { Button, Container, InputGroup, InputGroupText, Input, Label, Jumbotron } from 'reactstrap';
import './App.css';

class App extends Component {
  render() {
    return (
      <div className="App">
        <Jumbotron>
          <Container>
            <h1 className="display-3">Serverless Inference Demo</h1>
            <p className="lead">Try out text ranking and sentence classification models deployed on AWS Lambda</p>
            <p className="lead">
              <Button color="primary">Learn More</Button>
            </p>
          </Container>
        </Jumbotron>
        <div className="pt-3 pb-5 px-3">
          <Container>
            <div>
              <h2 class="display-4">SM-CNN</h2>
              <p class="lead">Short text pair ranking</p>
              <InputGroup>
                <Input placeholder="how are glacier caves formed" />
              </InputGroup>
              <br />
              <InputGroup>
                <Input placeholder="the ice facade is approximately 60 m high" />
              </InputGroup>
              <br />
              <Button className="float-right" color="secondary">Evaluate</Button>
            </div>
          </Container>
        </div>
        <div className="bg-light pt-3 pb-5 px-3">
          <Container>
            <div>
              <h2 class="display-4">Kim CNN</h2>
              <p class="lead">Sentence classification</p>
            </div>
          </Container>
        </div>
      </div>
    );
  }
}

export default App;
