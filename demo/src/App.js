import React, { Component } from 'react';
import { Container, Jumbotron, Button } from 'reactstrap';
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
      </div>
    );
  }
}

export default App;
