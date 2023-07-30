import React,{Component} from "react";
import Container from 'react-bootstrap/Container';
import Navbar from 'react-bootstrap/Navbar';
class Header extends Component{
    state = { }
    
    render() {
    const centerStyle = {
    textAlign: 'center',
    color: 'white',
    // Add any other custom styles you want
  };
       return (
      <Container>
    <Navbar bg="dark" expand="lg" variant='dark'>
      <Container>
        <h1  style={centerStyle} >MASCS:MULTI-AGENT SLAM FOR COLLABORTAIVE SEARCH AND RESCUE</h1>
         
      </Container>
    </Navbar>
    </Container>
  );
}
  


  
    }

export default Header;