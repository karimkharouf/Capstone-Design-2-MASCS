import React,{Component} from "react";
import { Alert } from "react-bootstrap";
class Message extends Component{
    state = {connected: false, ros:null,};
    constructor(){
        super();
        this.init_connection();
    }
    init_connection(){
         this.state.ros = new window.ROSLIB.Ros();
        this.state.ros.on("connection",()=>{
            console.log("connection established");
            this.setState({connected:true});
        });
        this.state.ros.on("close",()=>{
            console.log("connetion failed");
            this.setState({connected:false});
        });
        this.state.ros.connect("ws://192.168.26.203:9090")  
    }
    componentDidMount() {
    this.detect();
  }
  detect(){
    var detection = new window.ROSLIB.Topic({
      ros:this.state.ros,
      name:"/detected",
      messageType:"std_msgs/Bool"
    });
    detection.subscribe((message)=>{
    this.setState({connected:message.data});

    });
  }

    
    render(){
        return(
           <Alert className="text-center m-3" variant={this.state.connected?"success":"danger"}>
                {this.state.connected? "Human detected ": "No human detected"}
                </Alert>
        )
    }


}
export default Message;