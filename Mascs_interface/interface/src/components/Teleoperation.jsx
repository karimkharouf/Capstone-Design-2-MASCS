import React, {Component} from "react";
import { Joystick } from "react-joystick-component";
class Teleoperation extends Component{
    state = {ros:null }
    constructor(){
        super();
        this.init_connection();
        this.handleMove = this.handleMove.bind(this);
        this.handleStop = this.handleStop.bind(this);
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
    //tb0/cmd_vel
    handleMove(event){
       var cmd_vel = new window.ROSLIB.Topic({
        ros:this.state.ros,
        name: "/tb3_1/cmd_vel",
        messageType:"geometry_msgs/Twist"
       })
       var twist = new window.ROSLIB.Message({
        linear:{
            x:event.y,
            y:0,
            z:0,
        },
        angular:{
            x:0,
            y:0,
            z:-event.x, 
        },

       })
       cmd_vel.publish(twist);

    }
    handleStop(){
        var cmd_vel = new window.ROSLIB.Topic({
        ros:this.state.ros,
        name: "/tb3_1/cmd_vel",
        messageType:"geometry_msgs/Twist"
       })
       var twist = new window.ROSLIB.Message({
        linear:{
            x:0,
            y:0,
            z:0
        },
        angular:{
            x:0,
            y:0,
            z:0
        }

       })
       cmd_vel.publish(twist);
       

    }
    render(){
        return(
            <div>
               <Joystick
               size = {100}
                    baseColor = "black"
                    stickColor = "grey"
                    move = {this.handleMove}
                    stop = {this.handleStop}
               >
                    
                    
                </Joystick> 
            </div>
        )
    }
}
export default Teleoperation;