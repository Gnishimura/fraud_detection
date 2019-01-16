# fraud_detection
Detecting fraud for a clothing website

Challenge Description

Company XYZ is an e-commerce site that sells hand-made clothes. You have to build a model that predicts whether a user has a high probability of using the site toperform some illegal activity or not. You only have information about the user first transaction on the site and based on that youhave to make your classification ("fraud/no fraud"). 

These are the tasks you are asked to do: 
* For each user, determine her country based on the numeric IP address. 
* Build a model to predict whether an activity is fraudulent or not. 
* Explain how differentassumptions about the cost of false positives vs false negatives would impact the model. 

Your boss is a bit worried about using a model she doesn't understand for something asimportant as fraud detection. How would you explain her how the model is making the predictions? Not from a mathematical perspective (she couldn't care less about that), but from a user perspective. What kinds of users are more likely to be classified as at risk? What are their characteristics? Let's say you now have this model which can be used live to predict in real time if an activity is fraudulent or not. From a product perspective, how would you use it? That is, what kind of different user experiences would you build based on the model output?


Features:
* user_id : Id of the user. Unique by user
* signup_time : the time when the user created her account (GMT time)
* purchase_time : the time when the user bought the item (GMT time)
* purchase_value : the cost of the item purchased (USD)
* device_id : the device id. You can assume that it is unique by device. I.e.,  transactions with the same deviceID means that the same physical device was used to but 
* source : user marketing channel: ads, SEO, Direct (i.e. came to the site by directly typing the site address on the browser).
* browser : the browser used by the user.
* sex : user sex: Male/Female
* age : user age
* ip_address : user numeric ip address
* class : this is what we are trying to predict: whether the activity was fraudulent (1) or not
