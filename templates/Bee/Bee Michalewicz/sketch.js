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
let bumblebee;
let waggledance=true
let timer=0.5
let phase="Onlooker"
function preload(){
  table = loadTable('bee_Michalewicz.csv', 'csv','header');
  img=loadImage('Bee1.png')
bumblebee = loadImage('bumblebee.png');
}

function setup() {
  createCanvas(600, 600);
  createTable();
  initialize();
}
function draw() {
  frameRate(60)
 
  background(img);
  textAlign(LEFT, CENTER);
  stroke('black');
  strokeWeight(5)
  fill('rgb(255,37,0)')
  textSize(35);
  text(phase, 20, 40);
  if(waggledance){
    for (let j = 0; j < number_of_ind; j++) {
    let p = currentPositions[iter][j];
    p.x += random(-10, 10); // add some randomness to x position
    p.y += random(-10, 10); // add some randomness to y position
    ellipse(p.x, p.y); // draw the point
  }
  }
  addVelocity(iter);
  move(iter);
   if (frameCount % 30 == 0 && timer > 0) { 
    timer =timer-0.5;
  }
  if (timer == 0) {
    waggledance=false
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
      points[row.get('Id')]=[row.get('X'),row.get('Y'),p];
      
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
    currentPositions[i][j] = createVector(float(iterations[i][j])[0],float(iterations[i][j][1]),iterations[i][j][2]);
    
    targetPositions[i][j] = createVector(float(iterations[i+1][j][0]),float(iterations[i+1][j][1]),iterations[i+1][j][2]);
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

  for (let j = 0; j<number_of_ind; j++) {
    let currentPosition = currentPositions[i][j]
    strokeWeight(10)
    
    
    ellipse(image(bumblebee,currentPosition.x-35, currentPosition.y-25,30,30));}
  
  
  //targetPositions[i][j] = currentPosition.copy();
}
function mousePressed() {
 timer=0.5
 iter=iter+1;
 print(iter)
  if(iter%3==0){
    waggledance=true;
    phase="Onlooker"
  }
  if(iter%3==1){
    phase="Scout"
  }
  if(iter%3==2){
    phase="Employed"
  }
  
 if(iter>=3*number_of_iterations-1){
   iter=0
 }
}
