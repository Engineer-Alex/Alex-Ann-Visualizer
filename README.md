# Alex_A Neural Network MNIST Input Grid #1

**Author / Credit**  
Eng. Alex Amaral - AB&C Engineering Systems, LLC , and not small thanks to GitHub/Copilot Agent Assistance (those tools are the feature)
April 06 2026

## 1. What this project does

This small web application lets a student draw a digit on a `28 x 28` grid and then see two different ways a computer can try to recognize that digit:

1. a **real ANN model** trained with TensorFlow.js and MNIST data
2. a **rule-based baseline** that uses fixed feature checks and template matching

The goal is to help students **see the difference between machine learning and hand-written rules**.

---

## 2. Main learning goal

This demo was built to show that:

- an **Artificial Neural Network (ANN)** learns from example data
- a **rule-based system** does not learn; it follows rules written by a person
- both can give an answer, but they work in very different ways

---

## 3. How the app works

### Input side
- The student draws a number in the left canvas.
- The drawing is stored as a `28 x 28` matrix.
- Each square holds a value between `0` and `1`.

### ANN side
- The app trains a small neural network in the browser using **TensorFlow.js**.
- It uses the **MNIST** handwritten digit dataset.
- The ANN predicts which digit (`0` to `9`) is most likely.

### Rule-based side
- The app also looks at simple shape clues such as:
  - top stroke
  - middle stroke
  - bottom stroke
  - center column
  - left/right balance
- It compares the drawing against fixed digit templates.
- This gives a second, non-ANN answer for comparison.

---

## 4. Files in this folder

- `index.html` — page structure and labels
- `style.css` — visual design and layout
- `visualize_v1.js` — drawing logic, MNIST training, ANN visualization, and rule-based baseline logic

---

## 5. How to use it

1. Open the page in a browser.
2. Wait for the model to load and train.
3. Draw a digit in the left grid.
4. Watch the ANN and the rule-based baseline respond.
5. Use **Clear** to erase the drawing.
6. Use **Train / Reload MNIST Model** only when you want to reload or retrain the ANN model.

---

## 6. Important note for students

- The **top visualization** is a real ANN.
- The **lower rule-based demo** is **not** an ANN.
- It is included only to help compare a learned system with a hand-built one.

---

## 7. Suggested classroom discussion questions

- Why can two systems give the same answer for different reasons?
- Why does the ANN sometimes look less certain even when it is correct?
- Why is a rule-based system easier to explain, but usually less accurate?
- What does it mean for a model to “learn from data”?
