# LifeSignal – AI-Enhanced Health Monitoring iOS & Apple Watch App

LifeSignal is a prototype **iOS + Apple Watch health monitoring system** designed to support elderly users through real-time vital sign tracking and AI-assisted risk assessment.

This project focuses on **end-to-end product implementation**, including iOS app development, wearable integration, backend services, and an AI-powered health analysis pipeline.

---

## 📱 Key Features

- **iOS App (SwiftUI)** for user onboarding, health condition setup, and real-time health dashboards
- **Apple Watch App (WatchKit)** for wearable-side data collection and secure device pairing
- **HealthKit integration** for vital sign monitoring (heart rate, blood oxygen)
- **Remote pairing workflow** using a secure 6-digit code
- **Hybrid AI risk assessment system** combining rule-based logic and machine learning
- Backend services for data storage, analysis, and caregiver access

---

## 🧠 AI & System Design

The system uses a **hybrid AI approach**:
- A **rule-based clinical scoring system** to ensure interpretability and safety
- A **Gradient Boosting classifier** to enable adaptive learning and personalization
- A dynamic weighting mechanism that gradually increases ML influence as more user data becomes available

AI is integrated as part of the **product workflow**, supporting real-time insights rather than serving as a standalone demo.

---

## 🛠 Tech Stack

- **iOS / WatchOS:** SwiftUI, WatchKit, Xcode, HealthKit  
- **Backend:** Python, RESTful APIs, MongoDB  
- **Machine Learning:** scikit-learn, Gradient Boosting  
- **System Design:** Secure device pairing, real-time data pipeline

---

## 📸 App Screenshots & UI

Due to the academic nature of this project, a detailed **technical report** documents the system architecture, AI pipeline, and UI design.

📌 **iOS and Apple Watch app screenshots are included in the Appendix section of the project report.**

---

## 📄 Project Report

A comprehensive report describing:
- iOS & Watch app implementation
- Pairing workflow
- Backend architecture
- AI model design and evaluation

*(See Appendix for iOS and WatchOS UI screenshots)*

---

- This repository reflects an **academic prototype** and includes exploratory code, experiments, and model testing.
- The project emphasizes **product design and system integration** rather than production deployment.
