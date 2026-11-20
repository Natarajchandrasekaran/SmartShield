"""
SmartShield Dataset Generator
Generates a labeled dataset of Indian SMS scam + legitimate messages
for training the TinyBERT classifier (Stage 2).
"""

import csv
import os
import random

# ============================================================
# SCAM MESSAGES (8 categories)
# ============================================================

SCAM_MESSAGES = {
    "fake_kyc": [
        "Dear SBI customer, your KYC has expired. Update immediately or your account will be blocked. Click: http://sbi-kyc-update.xyz/verify",
        "URGENT: Your HDFC Bank KYC is pending. Complete verification within 24 hours to avoid account suspension: http://hdfc-kycverify.top/update",
        "Your PAN-Aadhaar linking is incomplete. Update now to continue banking services: http://pan-aadhaar-link.xyz",
        "Alert: Your Airtel KYC verification is expiring today. Re-verify here: http://airtel-kyc.xyz/renew",
        "Dear customer, your SBI YONO KYC has not been updated. Click to verify: http://sbi-yono-kyc.top",
        "ICICI Bank: Your account KYC needs immediate update. Failure will result in account freeze: http://icici-update.xyz",
        "Jio KYC expired. Your number will be deactivated in 24hrs. Update here: http://jio-kyc-update.in",
        "PNB Alert: Complete your re-KYC before account gets blocked. Visit: http://pnb-rekyc.top/update",
        "Your Paytm KYC is incomplete. Complete now or wallet will be blocked: http://paytm-kyc.xyz/verify",
        "Kotak Bank: KYC re-verification required urgently. Click: http://kotak-verify.xyz/kyc",
        "ALERT: Your Axis Bank KYC will expire tomorrow. Update immediately: http://axis-kyc-renew.top",
        "BOB customer your KYC is pending since long time update it now http://bob-kyc.xyz or account will freeze",
        "Union Bank mein aapka KYC pending hai. Turant update karein: http://unionbank-kyc.top",
        "Canara Bank KYC notice: Your account will be limited if KYC not completed by today. Link: http://canara-kyc.xyz",
        "Federal Bank: We noticed your KYC documents are outdated. Submit updated docs via: http://federal-kyc.top",
        "Your IDFC First Bank KYC needs immediate attention. Verify now: http://idfc-kyc.xyz/submit",
        "Dear Bandhan Bank customer, mandatory KYC update required. Click: http://bandhan-kyc.xyz",
        "South Indian Bank KYC alert: Complete eKYC online: http://sib-ekyc.top/verify",
        "Aapka KYC khatam ho gaya hai. Bank account band hone se pehle update karein: http://kyc-india.xyz",
        "IndusInd Bank: Your KYC re-verification is overdue. Update: http://indusind-kyc.top",
        "HDFC KYC update karo nahi to account block ho jayega link: http://hdfc-update.xyz",
        "Aapka PAN card link nahi hua hai Aadhaar se. Abhi karo: http://pan-link.xyz",
        "Dear customer aapka SBI account KYC expire ho chuka hai click kare http://sbi-kyc.top",
        "Your Yes Bank KYC verification is incomplete. Submit here: http://yesbank-kyc.xyz",
        "RBL Bank: KYC documents expired. Re-submit within 48hrs: http://rbl-verify.top",
    ],

    "upi_fraud": [
        "Rs.15,000 credited to your account. Approve the collect request to receive the amount in your UPI wallet.",
        "You have received Rs.10,000 via UPI. Accept the collect request to complete the transfer.",
        "Dear customer, Rs.5,000 is pending in your PhonePe account. Accept collect request to receive.",
        "BHIM UPI: Rs.25,000 transfer initiated to your account. Approve the collect request now.",
        "Google Pay: Rs.8,500 sent to you. Approve the pending collect request to receive funds.",
        "Paytm: Rs.12,000 cashback credited. Accept the collect request to add it to your wallet.",
        "PhonePe reward of Rs.2,000 is waiting. Accept collect request to claim your reward.",
        "Rs.50,000 credited from Income Tax refund. Approve UPI collect to receive in your bank account.",
        "Dear customer, approve the UPI collect request of Rs.1 to verify your account and receive Rs.5000 bonus.",
        "Aapke account mein Rs.20,000 aaya hai. Collect request accept karein paisa pane ke liye.",
        "Rs.3,000 ka cashback mila hai aapko. Collect request approve karein.",
        "NPCI: Your pending UPI transfer of Rs.7,500 requires approval. Accept now.",
        "Your GPay scratch card won Rs.1,00,000! Approve collect request to redeem prize.",
        "PhonePe: Collect request pending for Rs.15,000 refund. Accept to receive.",
        "UPI money received Rs.10000. To add in your bank approve collect request sent.",
        "Aapke UPI wallet mein Rs.5000 pending hai. Accept collect request abhi.",
        "Rs.2,500 cashback from Flipkart sale. Accept UPI collect to receive.",
        "Dear user Rs.4000 sent via UPI. Accept pending collect to receive amount.",
        "Paisa bheja gaya hai Rs.8000. Collect request accept karo receive karne ke liye.",
        "BHIM app se Rs.6000 transfer hua hai. Approve karo collect request.",
        "GPay reward Rs.500 pending. Accept collect request abhi claim karo.",
        "UPI refund of Rs.3500 is pending approval. Accept the request now.",
        "Rs.9000 ka transfer pending hai aapke liye. Collect request accept karein.",
        "PhonePe cashback Rs.1000 mila hai. Approve collect request.",
        "Aapko Rs.7000 UPI transfer mila hai. Accept karo collect request.",
    ],

    "otp_phishing": [
        "We detected suspicious login on your account. Share the OTP sent to your mobile for verification.",
        "Your HDFC net banking session expired. Enter the OTP sent to your phone to re-login: Reply with OTP.",
        "SBI security alert: Unusual transaction detected. Verify by sharing the OTP we just sent you.",
        "Dear customer, your account is under review. Please share your OTP to complete verification process.",
        "ICICI Bank: Account verification required. Share OTP received on your registered mobile.",
        "Alert! Someone tried to login to your account. Share OTP with our executive to block the transaction.",
        "PhonePe security: Share your OTP to verify your identity and prevent unauthorized access.",
        "Google Pay alert: Share the OTP sent to your number to confirm your account security.",
        "Aapke account mein suspicious activity hai. OTP share karein verification ke liye.",
        "Dear user, your Paytm wallet is at risk. Send us the OTP to secure your account.",
        "Amazon account verification: Please tell the OTP sent to your mobile to verify your order.",
        "URGENT: Share OTP sent to your phone to prevent account from being hacked.",
        "Flipkart: Your order verification needs OTP confirmation. Please share OTP received.",
        "RBI verification: Share OTP sent to your registered number for account security check.",
        "Aapka OTP batao humko account verify karne ke liye. Nahi to block ho jayega.",
        "OTP share karo jaldi apna account safe karne ke liye. Customer care bol raha hai.",
        "Please provide the 6 digit OTP for completing your bank verification process.",
        "Tell us the OTP you just received to process your refund of Rs.5000.",
        "Share OTP to verify your Aadhaar linked account. This is mandatory by RBI.",
        "Dear customer OTP de dijiye aapka account freeze hone wala hai.",
        "Verification ke liye OTP share karein turant. Aapka account suspend ho sakta hai.",
        "Amazon delivery OTP share karein to release your package held at warehouse.",
        "OTP batao nahi to aapka loan application reject ho jayega.",
        "Share OTP sent to your number for income tax refund processing.",
        "OTP de do jaldi aapke account se suspicious payment ho raha hai.",
    ],

    "prize_scam": [
        "Congratulations! Your Jio number has won Rs.25,00,000 in lucky draw! Claim your prize now: bit.ly/jio-prize-claim",
        "You have won Rs.50 lakhs in KBC lottery! Call 9876543210 to claim your prize.",
        "Dear Airtel user, your number selected for Rs.10,00,000 reward! Click: bit.ly/airtel-reward",
        "WINNER! Your mobile number won Rs.5,00,000 in Flipkart Big Billion Days lucky draw! Claim: bit.ly/fk-winner",
        "Congratulations! You won a Tata Nexon car in Amazon Great Indian Festival! Claim: bit.ly/amazon-car",
        "Dear customer, your Paytm number won Rs.1,00,000 cashback! Claim now: http://paytm-prize.xyz",
        "WhatsApp lottery winner! You won Rs.75,00,000. Send name and bank details to claim.",
        "TATA Motors lucky draw: Your number won Rs.15,00,000! Call 8765432109 immediately.",
        "Badhai ho! Aapka number lucky draw mein select hua hai Rs.10 lakh ka inaam! Call karein: 9876543210",
        "Samsung Galaxy S25 free! You won in our anniversary draw. Claim: bit.ly/samsung-free",
        "Reliance Jio: Congratulations! Rs.5,00,000 jeetne ke liye link pe click karein: bit.ly/jio-win",
        "Dear user you have been selected for Rs.25 lakh prize money. Deposit Rs.5000 processing fee to claim.",
        "IRCTC lucky draw winner! You won 2 free airline tickets + Rs.1,00,000 cash. Call 9988776655.",
        "Google lottery: Your email selected for $500,000 prize. Reply with full name and phone number.",
        "Flipkart: Aap jeet gaye Rs.50,000 ka voucher! Claim karne ke liye click karein: bit.ly/fk-voucher",
        "Myntra anniversary draw: You won Rs.25,000 shopping voucher! Claim: bit.ly/myntra-win",
        "Zomato lucky customer! Free food for 1 year + Rs.10,000 cash! Claim: bit.ly/zomato-lucky",
        "BSNL: Aapka number select hua hai Rs.3,00,000 inaam ke liye. Call 7654321098.",
        "Vi (Vodafone-Idea): Congratulations winner! Rs.8,00,000 prize. Claim: bit.ly/vi-prize",
        "PhonePe mega jackpot winner Rs.1 crore! Process claim: bit.ly/phonepe-crore",
        "Swiggy: You won unlimited free delivery for 1 year! Claim prize: bit.ly/swiggy-free",
        "Aapne Rs.2 lakh ka inaam jeeta hai. Processing fees Rs.2000 bhejein to claim.",
        "Big Bazaar lucky draw: Rs.50,000 shopping voucher won! Call 8899776655 to claim.",
        "HDFC credit card reward: You won Rs.5,00,000! Claim: bit.ly/hdfc-reward",
        "Meesho seller reward: Rs.1,00,000 bonus won! Claim now: bit.ly/meesho-win",
    ],

    "loan_extortion": [
        "URGENT: Pay Rs.5000 within 2 hours or your contacts list will be notified about your loan default.",
        "Final warning: Your loan EMI of Rs.3,500 is overdue. Pay now or recovery agent will visit your home.",
        "NOTICE: Rs.8,000 loan overdue. Your family and office contacts will be informed if not paid by today.",
        "Pay your pending loan of Rs.12,000 immediately. Legal action and FIR will be filed.",
        "Warning: Loan default of Rs.15,000. Your contacts have been notified. Pay now to stop: 9876543210",
        "Dear borrower, your loan is 30 days overdue. We will share your photo with contacts if not paid today.",
        "Aapka loan Rs.5000 pending hai. Abhi pay karo nahi to aapke contacts ko message jayega.",
        "Recovery notice: Rs.7,500 due. Your workplace will be contacted. Pay immediately: 8765432109",
        "LAST CHANCE: Pay Rs.4,000 loan now or we will file police complaint and inform your relatives.",
        "Aapke EMI ka payment pending hai. Family ko inform kiya jayega agar aaj payment nahi hua.",
        "Loan default notice: Rs.10,000. Your employer and family have been notified. Pay now.",
        "URGENT: Recovery agent dispatched to your address. Pay Rs.6,000 immediately to cancel visit.",
        "Warning: We have your contact list. Pay Rs.3,000 by 6 PM or messages will be sent.",
        "Aapka loan overdue hai. Contacts ko bheja jayega message. Turant Rs.5000 pay karein.",
        "Legal notice: Loan recovery of Rs.20,000. Court summons will be issued if not paid today.",
        "Pay karo Rs.8000 abhi nahi to aapki photo aapke contacts ko bhej denge.",
        "Loan default: Rs.4500 pending. Police complaint filed. Pay now to withdraw.",
        "Dear customer loan Rs.9000 overdue. Recovery process initiated. Pay: 7654321098.",
        "ALERT: Rs.6500 EMI bounce. CIBIL score affected. Pay now to avoid legal action.",
        "Aapke ghar recovery agent aa raha hai. Rs.3500 abhi pay karo cancel karne ke liye.",
        "Warning: Aapke office mein call kiya jayega loan default ke baare mein. Pay Rs.5000 now.",
        "Court notice: Loan amount Rs.15000 recovery. Appear in court or pay immediately.",
        "FIR file hone wali hai aapke khilaf. Loan Rs.7000 abhi pay karo.",
        "Final notice: Rs.11000 loan. Contacts, family, employer sab ko inform kiya jayega.",
        "Cyber crime complaint registered for loan default Rs.4000. Pay to settle.",
    ],

    "bank_impersonation": [
        "ALERT: Your HDFC Bank account will be blocked within 24 hours. Call 9876543210 immediately to verify.",
        "SBI: Your debit card has been temporarily blocked due to suspicious activity. Call 8765432109 to unblock.",
        "ICICI Bank security alert: Unauthorized login detected. Call our helpline 7654321098 to secure account.",
        "Dear Axis Bank customer, your account has been frozen. Contact 6543210987 to reactivate.",
        "Kotak Mahindra Bank: Your net banking access suspended. Verify identity: http://kotak-verify.xyz",
        "PNB alert: Rs.49,999 debit attempted from your account. Call 9988776655 to block.",
        "Your SBI account is under investigation for money laundering. Call CBI helpline 8877665544 immediately.",
        "HDFC Bank: Your credit card limit has been blocked. Update details: http://hdfc-card.xyz",
        "Dear BOB customer your account frozen by RBI order. Unfreeze by calling 7766554433.",
        "Canara Bank: Account suspended due to incomplete documentation. Call 6655443322.",
        "Yes Bank: Suspicious withdrawal of Rs.25,000 from your account. Call 5544332211 to report.",
        "Aapka SBI account block ho gaya hai. Turant call karein 9876543210 ya link pe click karein: http://sbi-verify.xyz",
        "Union Bank: Your ATM card cloned. Block immediately by calling 4433221100.",
        "Federal Bank alert: Account compromised. Secure now: http://federal-secure.xyz",
        "IndusInd Bank: Credit card fraud detected Rs.30,000. Call 3322110099 urgently.",
        "Dear IDFC customer account access revoked call 2211009988 for immediate assistance.",
        "RBL Bank: Account frozen RBI directive. Contact 1100998877 to resolve.",
        "Bandhan Bank: Your FD of Rs.5,00,000 at risk. Verify: http://bandhan-fd.xyz",
        "South Indian Bank: Unauthorized NEFT Rs.75,000. Report to 0099887766.",
        "Aapka HDFC account mein fraud transaction hua hai. Call karein 9876543210 abhi.",
        "ICICI credit card se Rs.45000 ka transaction hua. Agar aapne nahi kiya to call karein 8765432109.",
        "SBI se message: Account verify karein nahi to 48 ghante mein band ho jayega. Link: http://sbi-verify.top",
        "Axis Bank fraud alert: Debit card transaction Rs.20000. Not you? Call 7654321098.",
        "PNB mein aapka account suspicious hai. RBI investigation. Call 6543210987.",
        "Kotak se important notice: Account freeze hone wala hai. Update karein: http://kotak-update.xyz",
    ],

    "fake_delivery": [
        "Your Amazon parcel is held at customs. Pay Rs.49 delivery charges to release: http://amzn-delivery.in/pay",
        "Flipkart: Your order delivery failed. Reschedule by paying Rs.25 redelivery fee: http://fk-redeliver.xyz",
        "India Post: Parcel #IN7890 held due to unpaid customs duty of Rs.150. Pay here: http://indiapost-duty.top",
        "DTDC courier: Package cannot be delivered. Address verification needed. Click: http://dtdc-verify.xyz",
        "BlueDart: Your shipment is on hold. Pay Rs.99 processing fee: http://bluedart-fee.top",
        "Your international parcel requires customs clearance fee of Rs.500. Pay: http://customs-pay.xyz",
        "FedEx India: Delivery attempt failed. Confirm address and pay Rs.75: http://fedex-india.top",
        "Amazon: Order #AZ98765 delivery delayed. Update address: http://amzn-update.xyz",
        "Meesho order stuck at warehouse. Pay Rs.30 to release: http://meesho-ship.xyz",
        "Delhivery: Package returned to sender. Rebook for Rs.50: http://delhivery-rebook.top",
        "Aapka courier ruka hua hai customs mein. Rs.200 pay karein release ke liye: http://courier-pay.xyz",
        "Ekart: Delivery boy unable to reach. Reschedule: http://ekart-schedule.top",
        "Your Myntra order returned due to incomplete address. Re-order: http://myntra-reorder.xyz",
        "Speed Post: Article held for Rs.100 postage due. Pay online: http://speedpost-pay.top",
        "Snapdeal: Order delivery failed 3 times. Final attempt after payment of Rs.45: http://snap-deliver.xyz",
        "Professional Courier: Package damaged in transit. Insurance claim: http://procourier.xyz",
        "Aapka Amazon parcel customs mein hai. Rs.49 charges pay karein: http://amzn-customs.xyz",
        "Flipkart delivery failed. Pay Rs.35 for redelivery: http://fk-pay.top",
        "Courier charges pending Rs.60. Pay karein nahi to parcel wapas chala jayega: http://courier.xyz",
        "DTDC: Parcel mein customs issue hai. Rs.100 pay karo: http://dtdc-customs.top",
        "BlueDart shipment pe duty Rs.75 pending hai. Pay karein: http://bd-duty.xyz",
        "India Post parcel duty Rs.120 pending. Online payment karein: http://post-duty.top",
        "Delivery boy waiting at your door. Pay COD charges Rs.199 online: http://cod-pay.xyz",
        "Meesho return pickup failed. Pay Rs.40 rescheduling charge: http://meesho-return.xyz",
        "Ajio order customs hold. Release payment Rs.80: http://ajio-customs.top",
    ],

    "job_scam": [
        "Earn Rs.5000-10000 daily from home! Simple typing job. WhatsApp us: 9876543210. Limited seats!",
        "Work from home opportunity! Earn Rs.3000/day doing simple online tasks. Join: bit.ly/home-job",
        "Part time job: Earn Rs.1500-5000 per day. No experience needed. WhatsApp: 8765432109",
        "Dear candidate, selected for data entry job. Rs.8000/day. Start immediately: bit.ly/data-entry-job",
        "Amazon/Flipkart product review job. Earn Rs.2000 per review. Contact: 7654321098",
        "YouTube like/subscribe job. Earn Rs.500 per task. Daily payment. WhatsApp: 6543210987",
        "Instagram followers job: Rs.100 per follow. Earn Rs.5000+ daily. Join: bit.ly/insta-job",
        "Telegram task job: Complete simple tasks, earn Rs.3000 daily. Join channel: t.me/easymoney",
        "Ghar baithe kamai karein Rs.10000 daily! Simple mobile se kaam. WhatsApp: 9876543210",
        "Copy paste job from home. Rs.500 per page. Unlimited work. Contact: 8765432109",
        "Online survey job: Earn Rs.1000 per survey. 10 surveys daily. Register: bit.ly/survey-earn",
        "Freelance content writing: Rs.5000 per article. No experience. WhatsApp: 7654321098",
        "Mobile recharge job: Earn commission Rs.200 per recharge. Join: bit.ly/recharge-earn",
        "Dear student, earn Rs.15000/month part time. Simple form filling. WhatsApp: 6543210987",
        "Investment job: Invest Rs.1000 earn Rs.5000 daily return. Guaranteed. WhatsApp: 9988776655",
        "Crypto trading bot: Invest Rs.500 earn Rs.10000 daily. Join: bit.ly/crypto-earn",
        "Ad watching job: Rs.50 per ad. Watch 100 ads daily = Rs.5000. App: bit.ly/watch-earn",
        "SMS sending job from mobile. Rs.3 per SMS. Send 1000 SMS = Rs.3000 daily. Call: 8877665544",
        "Captcha typing job: Rs.5 per captcha. Unlimited work. Register: bit.ly/captcha-job",
        "App testing job: Test apps earn Rs.500 per test. Daily 10 apps. WhatsApp: 7766554433",
        "Data entry kaam ghar se. Rs.8000 daily income. Abhi join karein: bit.ly/data-work",
        "Simple task karo Rs.2000 roz kamao. Mobile se kaam. WhatsApp: 6655443322",
        "Paisa kamao daily Rs.5000. Bas WhatsApp pe message karo: 5544332211",
        "Online job Rs.3000 daily guaranteed. Limited time offer. Join: bit.ly/quick-money",
        "Part time typing kaam. Rs.1000 per page. Unlimited pages. WhatsApp: 4433221100",
    ],
}


# ============================================================
# LEGITIMATE MESSAGES
# ============================================================

LEGITIMATE_MESSAGES = [
    # --- Genuine Bank Transactions ---
    ("INR 2,500.00 debited from A/c XX4532 on 15-Nov-2026. UPI Ref No: 433298765432. If not done by you, call 1800-111-109. -SBI", "legitimate"),
    ("INR 15,000.00 credited to A/c XX8976 by NEFT. Ref: HDFC26112000001. Balance: INR 45,230.50. -HDFC Bank", "legitimate"),
    ("Rs.1,200.00 debited from A/c XX3421 at AMAZON PAY. UPI Ref: 433198765000. Avl Bal: Rs.8,750.00. -ICICI Bank", "legitimate"),
    ("INR 500.00 received from XXXXX@upi in your A/c XX7891. UPI Ref No: 433298765000. -HDFC Bank", "legitimate"),
    ("Credit Card XX9087 payment of Rs.12,500.00 received. Thank you. -Axis Bank", "legitimate"),
    ("EMI of Rs.8,333.00 debited from A/c XX5678 for Home Loan. Next EMI due: 15-Dec-2026. -SBI", "legitimate"),
    ("FD of Rs.1,00,000 booked for 1 year at 7.1% p.a. Maturity: 15-Nov-2027. Ref: FD987654. -HDFC Bank", "legitimate"),
    ("Your SBI debit card ending 4532 has been successfully activated. -SBI", "legitimate"),
    ("Salary of INR 35,000.00 credited to A/c XX2345 by PSG COLLEGE OF TECHNOLOGY. -ICICI Bank", "legitimate"),
    ("Rs.3,000.00 transferred to XXXXX@upi from A/c XX6789. UPI Ref: 433398765111. -Kotak Bank", "legitimate"),
    ("INR 750.00 debited for electricity bill payment. TNEB. UPI Ref: 433498765222. -Axis Bank", "legitimate"),
    ("Rs.199.00 debited for Netflix subscription from A/c XX1234. -HDFC Bank", "legitimate"),
    ("INR 5,000.00 transferred to A/c XX9876 via IMPS. IMPS Ref: 426398765333. -PNB", "legitimate"),
    ("Credit card statement for Nov 2026 generated. Total due: Rs.8,750.00. Due date: 15-Dec-2026. -SBI Card", "legitimate"),
    ("Mutual Fund SIP of Rs.5,000.00 debited from A/c XX3456 for HDFC Equity Fund. -HDFC Bank", "legitimate"),
    ("PPF deposit of Rs.10,000.00 credited to your PPF account. Balance: Rs.1,45,000.00. -SBI", "legitimate"),
    ("Rs.2,000.00 auto-debit for LIC premium from A/c XX7890. Policy: 12345678. -ICICI Bank", "legitimate"),
    ("INR 450.00 debited at ZOMATO from A/c XX4567. UPI Ref: 433598765444. -Axis Bank", "legitimate"),
    ("Your HDFC Bank savings account interest of Rs.1,230.00 credited for Q3 FY27. -HDFC Bank", "legitimate"),
    ("Rs.25,000.00 RTGS transfer successful to A/c XX8765. Ref: RTGS26112000005. -SBI", "legitimate"),

    # --- Genuine OTPs ---
    ("Your OTP for SBI Net Banking login is 847263. Valid for 5 minutes. Do not share this OTP with anyone. -SBI", "legitimate"),
    ("OTP for HDFC MobileBanking transaction is 592147. Valid for 3 minutes. Never share OTP with anyone. -HDFC", "legitimate"),
    ("Your one-time password for Paytm login is 438291. It expires in 10 minutes. Do not share with anyone.", "legitimate"),
    ("Google verification code: 734829. Don't share this code with anyone.", "legitimate"),
    ("Your Amazon OTP is 283947. Use it to verify your order. Do not share this with anyone.", "legitimate"),
    ("PhonePe OTP: 619283. Valid for 5 mins. Never share OTP. If you didn't request this, call 080-68727374.", "legitimate"),
    ("IRCTC OTP for booking: 847291. Valid 10 mins. Do not share.", "legitimate"),
    ("Flipkart login OTP: 392841. Valid for 5 minutes. Do not share with anyone.", "legitimate"),
    ("WhatsApp code: 829-374. You can also tap on this link to verify: v.whatsapp.com/829374", "legitimate"),
    ("Your Swiggy verification code is 584736. This code expires in 5 minutes.", "legitimate"),
    ("Uber OTP: 7384. Use this to verify your ride. Do not share with your driver.", "legitimate"),
    ("Ola verification code: 4829. Valid for 3 minutes.", "legitimate"),
    ("Myntra OTP: 639281. For login verification. Do not share.", "legitimate"),
    ("CRED verification code: 847291. Expires in 5 min. Never share with anyone.", "legitimate"),
    ("Groww OTP: 294837. For mutual fund transaction verification. Do not share.", "legitimate"),
    ("Zerodha Kite login OTP: 583921. Valid 5 min. Do not share.", "legitimate"),
    ("HDFC Securities OTP: 729384. For trade confirmation. Never share.", "legitimate"),
    ("NPCI: OTP for UPI registration is 847263. Valid 10 min. Do not share with anyone.", "legitimate"),
    ("DigiLocker OTP: 384729. For document access. Valid 5 min. Do not share.", "legitimate"),
    ("Aadhaar OTP: 827394. For eKYC verification at SBI branch. Do not share. -UIDAI", "legitimate"),

    # --- Genuine UPI Notifications ---
    ("Rs.500.00 received from XXXXX@upi in your A/c XX7891. UPI Ref No: 433298765000. -HDFC Bank", "legitimate"),
    ("Rs.1,500.00 sent to XXXXX@ybl from A/c XX4532. UPI Ref: 433198765111. -SBI", "legitimate"),
    ("UPI AutoPay of Rs.299.00 debited for Hotstar Premium. Mandate: MAN98765. -ICICI Bank", "legitimate"),
    ("Rs.2,000.00 received via UPI from COMPANY@axisb. Ref: 433398765222. -Axis Bank", "legitimate"),
    ("UPI Lite: Rs.50.00 paid to MERCHANT. Lite balance: Rs.450.00. -SBI", "legitimate"),

    # --- Genuine Delivery Updates ---
    ("Your Flipkart order #OD432876 has been shipped via Ekart. Track: https://www.flipkart.com/track/OD432876", "legitimate"),
    ("Amazon: Your order #402-7654321-8901234 is out for delivery today. Track on Amazon app.", "legitimate"),
    ("Myntra: Your order MN98765432 delivered successfully. Rate your experience on the app.", "legitimate"),
    ("Swiggy: Your order #SW9988 is out for delivery. Estimated arrival: 25 mins. Track on Swiggy app.", "legitimate"),
    ("Zomato: Your food order is being prepared. ETA: 35 minutes. Track live on Zomato app.", "legitimate"),
    ("Meesho: Order #MS876543 shipped. Track: https://www.meesho.com/order/MS876543", "legitimate"),
    ("DTDC: Your courier AWB 12345678 is out for delivery today. Track: https://www.dtdc.in/tracking", "legitimate"),
    ("BlueDart: Shipment 98765432 delivered to COIMBATORE. Received by: SELF. -BlueDart Express", "legitimate"),
    ("India Post: Speed Post EE123456789IN delivered at COIMBATORE HO on 15-Nov-2026.", "legitimate"),
    ("Delhivery: Package 123456789 is in transit. Expected delivery: 17-Nov-2026.", "legitimate"),

    # --- Genuine KYC / Official Reminders ---
    ("Reminder: As per RBI guidelines, please complete your KYC at the nearest SBI branch before 31-Dec-2026. -SBI", "legitimate"),
    ("HDFC Bank: Your KYC is due for renewal. Visit your home branch with PAN and Aadhaar. -HDFC Bank", "legitimate"),
    ("SEBI advisory: Link PAN with Aadhaar before 30-Jun-2027 for continued demat operations. -CDSL", "legitimate"),
    ("Income Tax Department: File your ITR for AY 2026-27 before 31-Jul-2026 on incometax.gov.in", "legitimate"),
    ("EPFO: Your PF passbook updated. Total balance: Rs.3,45,678.00. Check on epfindia.gov.in", "legitimate"),
    ("LIC: Premium due for Policy 12345678. Amount: Rs.8,500. Due date: 28-Nov-2026. Pay on licindia.in", "legitimate"),
    ("TNEB: Electricity bill for Nov 2026. Amount: Rs.1,250. Due date: 25-Nov-2026. Pay on tnebnet.org", "legitimate"),
    ("BSNL: Your broadband bill of Rs.799 is due on 20-Nov-2026. Pay on portal.bsnl.in", "legitimate"),

    # --- Genuine Promotional ---
    ("Flat 20% off on all Myntra orders this weekend! Use code WEEKEND20. T&C apply. -Myntra", "legitimate"),
    ("Amazon Great Indian Festival starts 15-Nov! Up to 80% off on electronics. Shop on Amazon app.", "legitimate"),
    ("Swiggy: 50% off up to Rs.100 on your next order. Code: SWIGGY50. Valid till midnight.", "legitimate"),
    ("CRED: Earn 5X reward points on credit card bill payment this month. Pay on CRED app.", "legitimate"),
    ("Zomato Gold: Free delivery + 20% off. Subscribe for Rs.149/month. Limited time offer.", "legitimate"),
    ("BigBasket: Fresh vegetables at 30% off. Order above Rs.500 for free delivery. -BigBasket", "legitimate"),
    ("PhonePe: Cashback of Rs.50 on recharge above Rs.199. Use code RECHARGE50. -PhonePe", "legitimate"),
    ("Google Pay: Earn up to Rs.501 scratch card on UPI payments this week. T&C apply.", "legitimate"),
    ("Flipkart: Electronics sale! Up to 70% off on smartphones, laptops. 12-18 Nov. -Flipkart", "legitimate"),
    ("BookMyShow: Avengers new movie tickets now available! Book on the app. -BookMyShow", "legitimate"),

    # --- Miscellaneous Legitimate ---
    ("Your SBI locker annual rent of Rs.3,000 is due. Visit branch to renew. -SBI", "legitimate"),
    ("IRCTC: PNR 2345678901 CNF. Train 12243 SBC RAJ, 15-Nov, Coach S5 Berth 23. -IRCTC", "legitimate"),
    ("Aarogya Setu: Your COVID vaccination certificate is available on CoWIN portal.", "legitimate"),
    ("DigiLocker: Your driving license has been successfully linked. Access on digilocker.gov.in", "legitimate"),
    ("Airtel: Your prepaid recharge of Rs.299 is successful. Validity: 28 days. Data: 1.5GB/day.", "legitimate"),
    ("Jio: Recharge successful. Plan Rs.239. Validity 28 days. 2GB/day + unlimited calls. -Jio", "legitimate"),
    ("Vi: Recharge of Rs.269 successful. Validity: 28 days. Data: 1GB/day. -Vi", "legitimate"),
    ("HDFC Ergo: Your car insurance policy renewed. Policy No: 12345678. Valid till 15-Nov-2027.", "legitimate"),
    ("Star Health: Health insurance premium of Rs.15,000 received. Policy active till 31-Mar-2027.", "legitimate"),
    ("KSRTC: Booking confirmed. PNR: BK12345. Coimbatore to Chennai, 16-Nov, Seat 23A.", "legitimate"),
]


def generate_dataset(output_dir: str):
    """Generate train/val/test CSV splits."""
    
    all_data = []
    
    # Add scam messages
    for category, messages in SCAM_MESSAGES.items():
        for msg in messages:
            all_data.append({
                "text": msg,
                "label": "scam",
                "category": category
            })
    
    # Add legitimate messages
    for msg, label in LEGITIMATE_MESSAGES:
        all_data.append({
            "text": msg,
            "label": label,
            "category": "legitimate"
        })
    
    # Shuffle
    random.seed(42)
    random.shuffle(all_data)
    
    total = len(all_data)
    train_end = int(total * 0.7)
    val_end = int(total * 0.85)
    
    splits = {
        "train": all_data[:train_end],
        "val": all_data[train_end:val_end],
        "test": all_data[val_end:]
    }
    
    for split_name, data in splits.items():
        filepath = os.path.join(output_dir, f"{split_name}.csv")
        with open(filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["text", "label", "category"])
            writer.writeheader()
            writer.writerows(data)
        
        scam_count = sum(1 for d in data if d["label"] == "scam")
        legit_count = sum(1 for d in data if d["label"] == "legitimate")
        print(f"  {split_name}.csv: {len(data)} messages "
              f"(scam: {scam_count}, legitimate: {legit_count})")
    
    print(f"\n  Total: {total} messages")
    print(f"  Scam categories: {len(SCAM_MESSAGES)}")
    for cat, msgs in SCAM_MESSAGES.items():
        print(f"    - {cat}: {len(msgs)} messages")
    print(f"  Legitimate: {len(LEGITIMATE_MESSAGES)} messages")


if __name__ == "__main__":
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 60)
    print("SmartShield Dataset Generator")
    print("=" * 60)
    generate_dataset(output_dir)
    print("\nDataset generated successfully!")
