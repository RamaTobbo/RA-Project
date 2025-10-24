let table;
let iterations = [];
let first_row;
let number_of_iterations;
let number_of_ind;
let number_of_rows;
let img;
let currentPositions=[];
let targetPositions =[];
let iter=0;
let firstfish
let phase="Colective Instictive\nMovement"
let swich=true
let counter=0

function preload(){
  // table = loadTable('fish_mic.csv', 'csv','header');
  // img=loadImage('Wolf1.png')
  // firstfish = loadImage('firstfish.png');
  // secondfish=loadImage('secondfish.png');
    table = loadTable('/static/fish_mic.csv', 'csv', 'header');
       img = loadImage('/static/assets/images/Wolf1.png');
   firstfish = loadImage('/static/assets/images/firstfish.png');
 secondfish = loadImage('/static/assets/images/secondfish.png');
}

function setup() {
  createCanvas(600, 600);
   createTable();
  initialize();
}

function draw() {
  background(img);
  frameRate(60)
  textAlign(LEFT, CENTER);
  stroke('black');
  strokeWeight(4)
  fill('rgb(246,20,93)')
  textSize(33);
  text(phase, 20, 40);
  addVelocity(iter);
  move(iter);
  if (frameCount % 10 == 0) { 
    counter =counter+1;
  }
  if (counter%2 == 0) {
    swich=false
  }
  else{
    swich=true
  }
  
}

function createTable(){
  //getting the first row
  first_row=table.getRow(0);
  number_of_rows=table.rowCount;
   
  //getting the nb of iterations and bees
  number_of_iterations=first_row.get('Iteration');
  number_of_ind=first_row.get('Id');
 
  for (let i =0;i<number_of_iterations;i++){
    for(let p=0;p<3;p++){
    points=[]
    for (let j= 0; j<number_of_ind;j++){
      
      row=table.getRow((number_of_ind*(3*i+p)+j+1));
      points[row.get('Id')]=[row.get('X'),row.get('Y'),row.get('Weight'),p];
      
    }
    iterations[3*i+p]=points;
    }
    
}
 print(iterations)
}
function initialize(){
  
    for (let i = 0; i < 3*number_of_iterations - 1; i++) {
    
  currentPositions[i] = [];
  targetPositions[i] = [];
}
  
   for (let i =0;i<3*number_of_iterations-1;i++){
      
  for (let j = 0; j< number_of_ind; j++) {
    currentPositions[i][j] = createVector(float(iterations[i][j])[0],float(iterations[i][j][1]),float(iterations[i][j][2]));
    
    targetPositions[i][j] = createVector(float(iterations[i+1][j][0]),float(iterations[i+1][j][1]),float(iterations[i+1][j][2]));
  }}
  print(currentPositions)
}
function addVelocity(i){
 
  
  for (let j = 0; j< number_of_ind; j++) {
    
    currentPosition = currentPositions[i][j]
    
    targetPosition = targetPositions[i][j]
    let velocity = targetPosition.copy().sub(currentPosition).mult(0.05);
    currentPosition.add(velocity);
    //targetPositions[i][j] = currentPosition.copy();
  }}
function move(i){
     //fill(255, 0, 0);
if(swich){
  for (let j = 0; j<number_of_ind; j++) {
    let currentPosition = currentPositions[i][j]
    strokeWeight(10)
    
    
    ellipse(image(firstfish,currentPosition.x-(currentPosition.z*2)/2, currentPosition.y-(currentPosition.z*3)/2,currentPosition.z*2,currentPosition.z*3));}
}
  else{
    for (let j = 0; j<number_of_ind; j++) {
    let currentPosition = currentPositions[i][j]
    strokeWeight(10)
    
    
    ellipse(image(secondfish,currentPosition.x-(currentPosition.z*2)/2, currentPosition.y-(currentPosition.z*3)/2,currentPosition.z*2,currentPosition.z*3));}
}
  
  
  //targetPositions[i][j] = currentPosition.copy();
}
function mousePressed() {
 iter=iter+1;
 print(iter)
  if(iter%3==0){
    phase="Colective Instictive\nMovement"
  }
  if(iter%3==1){
    phase="Colective Volitive\nMovement"
  }
  if(iter%3==2){
    phase="Individual \nMovement"
  }
  
 if(iter>=3*number_of_iterations-1){
   iter=0
 }
}