import Container from 'react-bootstrap/Container';
import Nav from 'react-bootstrap/Nav';
import Navbar from 'react-bootstrap/Navbar';
import NavDropdown from 'react-bootstrap/NavDropdown';
import {Route, BrowserRouter as Router,Switch} from "react-router-dom"
import Home from "./Home";
import About from "./About";
import React,{Component} from 'react';
import Connection from './Connection'
import Teleoperation from './Teleoperation';
import RobotState from './RobotState';
import RobotState2 from './RobotState2';
import { Col, Row } from 'react-bootstrap';
import Camera from './Camera';
import Teleoperation2 from './Teleoperation2';
import Map from './map';
import Message from "./message";

  

class Body extends Component {
    state = {};
    render(){

    
  return (
    <div>
      <Container>
        <h1 className='text-center mt-3'>Robot control</h1>
        <Row>
          <Col>
          <Connection/>
          </Col>
        </Row>
        <Row>
          <Col>
          <h3>Robot 1</h3>
        <Teleoperation />
          </Col>
          <Col></Col>
           <Col>
           <h3>Robot 2</h3>
          <Teleoperation2/>
          </Col> 
         
          <Col>
          </Col>
          
        </Row>
        <Row>
          {" "}
          
          <Col>
          <RobotState/> 
          </Col>
           <Col>
          <RobotState2/>
          </Col> 
          </Row>

          <Row>
          </Row>

          
           <Col>
           <Camera/>
           </Col>
           <Col>
           <Map/>
           </Col>
         
          
        
        <Row></Row>
        <Row></Row>
                  
          
           
        </Container>

    </div>
    
  );
}
}
export default Body;