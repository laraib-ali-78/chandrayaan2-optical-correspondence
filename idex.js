function Student(name, age) {
    this.name = name;
    this.age = age;
}

let students = [
    new Student("Ali", 20),
    new Student("Ahmed", 21),
    new Student("siddhant", 19)
];

console.log(students);

console.log(students);

let name1="John";
const name2="Inzmam";


console.log("before updation")
console.log(name1);
console.log(name2);


name1="rishikesh";//let updates value

//name2="alok";//const does not update value

console.log("after updation")
console.log(name1);
console.log(name2);

//object
let obj={name: "John", age:30, rollno:101};

console.log(obj.age);
console.log(obj.name);
console.log(obj.rollno);