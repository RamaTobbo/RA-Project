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
let bat;

function preload(){
  table = loadTable('bat_Mich.csv', 'csv','header');
  img=loadImage('Bat1.png')
  bat = loadImage('bat-logo-and-symbols-template-vector-21489239-removebg-preview.png');
}

function setup() {
  createCanvas(600, 600);
  createTable();
  initialize();
}

function draw() {
  background(img);
  
  addVelocity(iter);
  move(iter);
  

}
function createTable(){
  //getting the first row
  first_row=table.getRow(0);
  number_of_rows=table.rowCount;
   
  //getting the nb of iterations and bees
  number_of_iterations=first_row.get('Iteration');
  number_of_ind=first_row.get('Id');
 
  for (let i =0;i<number_of_iterations;i++){
    points=[]
    for (let j= 0; j<number_of_ind;j++){
      if(j<3){
   
      row=table.getRow((number_of_ind*i)+j+1);
      points[row.get('Id')]=[row.get('X'),row.get('Y')];  
      }
      else {
      row=table.getRow((number_of_ind*i)+j+1);
      points[row.get('Id')]=[row.get('X'),row.get('Y')];
      }
    }
    iterations[i]=points;
}
  print(iterations)
}
function initialize(){
    for (let i = 0; i < number_of_iterations - 1; i++) {
  currentPositions[i] = [];
  targetPositions[i] = [];
}
   for (let i =0;i<number_of_iterations-1;i++){
  for (let j = 0; j< number_of_ind; j++) {
    currentPositions[i][j] = createVector(float(iterations[i][j])[0],float(iterations[i][j][1]),iterations[i][j][2]);
   
    targetPositions[i][j] = createVector(float(iterations[i+1][j][0]),float(iterations[i+1][j][1]),iterations[i][j][2]);
  }}
 
}
function addVelocity(i){
 
 
  for (let j = 0; j< number_of_ind; j++) {
   
    currentPosition = currentPositions[i][j]
   
    targetPosition = targetPositions[i][j]
    let velocity = targetPosition.copy().sub(currentPosition).mult(0.05);
    currentPosition.add(velocity);
    //targetPositions[i][j] = currentPosition.copy();
  }
 
}
function move(i){
     //fill(255, 0, 0);

  for (let j = 0; j< number_of_ind; j++) {
    let currentPosition = currentPositions[i][j]
    strokeWeight(10)
    a=iterations[i][j][2]
    ellipse(image(bat,currentPosition.x-35, currentPosition.y-25,50,50));}
    }

function mousePressed() {
 iter=iter+1;
  
 if(iter>=number_of_iterations-1){
   iter=0
 }
}