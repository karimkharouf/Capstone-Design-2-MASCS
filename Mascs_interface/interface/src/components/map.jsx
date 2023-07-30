import React, { Component } from "react";

class Map extends Component {
  state = {
    ros: null,
    detectedPose:null,
    x: 0,
    y: 0,
  };

  constructor() {
    super();
    //this.init_connection = this.view_map.bind(this);
    this.view_map = this.view_map.bind(this);

  }

  init_connection() {
    //this.setState({ ros: new window.ROSLIB.Ros() });
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
    this.init_connection();
    //console.log("Map: componentDidMount" + this.state.ros);
    this.view_map();

  }

  view_map() {
    var viewer = new window.ROS2D.Viewer({
      divID: "map",
      width: 640,
      height: 480,
    });

   
    var navClient = new window.NAV2D.OccupancyGridClientNav({
      ros: this.state.ros,
      rootObject: viewer.scene,
      viewer: viewer,
      serverName: "/move_base",
      //map_topic:"/tb3_0/map",
      continuous:true,
      robot_pose: "/tb3_1/robot_pose",
      withOrientation: true
    });
     var detection = new window.ROSLIB.Topic({
      ros:this.state.ros,
      name:"/detected",
      messageType:"std_msgs/Bool"
    });
    detection.subscribe((message)=>{
      if(message.data===true){
        this.setState({detectedPose:navClient.robot_pose});
      }

    });
    
    var robotMarker = new window.ROS2D.TraceShape({
      pose: this.state.detectedPose
    });
    viewer.addObject(robotMarker);
  }
  

  render() {
    
    return (
        <div>
          <h3>MAP:</h3>
        <div id="map"></div>
        </div>
    );
  }
}

export default Map;