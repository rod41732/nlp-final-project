import React, { useState } from 'react';
import './App.css';
import { Button, Container, Form, Row } from "react-bootstrap";
import * as request from 'superagent';

function App() {
  const [result, setResult] = useState<string>('');
  const [text, setText] = useState<string>('');

  const submit = () => {
    request.post('https://nlp-api-bcrjpew6hq-as.a.run.app/predict').send({
      'text': text,
    }).then(res => {
      setResult(res.body.split(" ").join(""));
    })
    setResult("dummy")
  }

  return (
    <Container style={{maxWidth: '768px', paddingTop: 32}}>
      <Row className="justify-content-center">
        <h1> News Title Generator</h1>
      </Row>
      <Form>
        <Form.Group>
          <Form.Label> Enter News Content </Form.Label>
          <Form.Control value={text} onChange={(e) => {
            console.log(e.target.value);
            setText(e.target.value);
          }} as="textarea" rows={10}/>
        </Form.Group>
      </Form>
      <Button variant="primary" onClick={submit}> Generate Title ! </Button>
      <div style={{paddingTop: 16}}>
        <h2> Result </h2>
        {
          result ? (
            <div> { result } </div>
          ) : (
            <div> Please enter text and click generate </div>
          )
        }
      </div>
    </Container>
  );
}

export default App;
