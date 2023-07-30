import React,{Component} from "react";
import { Alert } from "react-bootstrap";
class Connection extends Component{
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
    
    render(){
        return(
            <div>
                <Alert className="text-center m-3" variant={this.state.connected?"success":"danger"}>
                {this.state.connected? "Robot Connected": "Robot Disconnected"}
                </Alert>
            </div>
        )
    }


}
export default Connection;