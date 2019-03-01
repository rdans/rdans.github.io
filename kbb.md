
# WebTesting - E2E
Main purpose of creating this project is to create a basic automation test to validate basic functionality of Kelley Blue Book website. It has 3 test scenarios:
1. E2E test, where users navigate to the website and try to get a value for users' car.
2. Test invalid login functionality where users:
2.1 Enter a wrong credentials.
2.2 Leave password or email field black.
<br />
<center>
<img align="middle" width="500" height="200" src="https://drive.google.com/uc?id=1Xd-tslIFM2VizJnEDn37DN9GujUzkRFM">
</center>


## Project Details
This project was written in Java with help of Selenium Web Driver, written in Page Object Model (POM) design pattern to enhance the flexibility of modifying the code, once the UI of the website changes. 
<br />
With help of Cucumber, it will improve the readability of the code by describing the use case scenario. For example, 
```Gherkin
Feature: Login Application
Scenario Outline: User failed to login

Given Initialize browser with chrome
And User navigate to "https://www.kbb.com/" site
And Click on login link in home page, click sign in button, and user will be at sign in page
When User enters "<username>" and "<password>" and sign in
Then Website will display "<error>" message
And close browser

Examples:

|username  |password  |error  |

|rei@gmail.com  |1234  |The email or password you've entered is not valid  |

| | |Missing required parameter  |
```   
 
The test also run in TestNG using Maven, which will give a test output scenario in ExtentReporterNG format. 

For full demo, you can watch the demo by click the image below

[![](http://img.youtube.com/vi/qYCp0PRo2k8/0.jpg)](http://www.youtube.com/watch?v=qYCp0PRo2k8 "e2e") 
 
 To see the project details, please go to the project [repositories](https://github.com/rdans/kbb) 
 
 [<< Home](README.md) 
 
### Contact me:
:email:
[reinaldo.daniswara@gmail.com](mailto:reinaldo.daniswara@gmail.com)


