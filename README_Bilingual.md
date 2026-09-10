# SmartSheet — Bilingual Competition Documentation Guide
# راهنمای دو زبانه مستندسازی مسابقه SmartSheet

## 1. What was corrected / چه چیزهایی اصلاح شد؟

The previous documentation mixed different evidence levels and contained several competition-risky claims. This version fixes them by:

- Separating the historical 95.2% controlled benchmark from the later six-scenario stress test.
- Reporting the newer stress-test weighted result as 71.7% and showing its failure modes.
- Removing unsupported claims such as universal smartphone compatibility or guaranteed real-world accuracy.
- Clarifying that the current core is deterministic OpenCV computer vision, not a trained deep-learning model.
- Removing placeholder GitHub/repository claims from the installation section.
- Making the project structure conditional on files actually present in the final ZIP.
- Explaining that confidence is a decision-support score, not a calibrated probability.
- Positioning uncertainty handling and human review as part of the educational value.
- Making the English version the primary competition-facing document while retaining Persian as a reference.

## 2. Recommended competition message / پیام پیشنهادی

**EN**

> SmartSheet uses computer vision to turn ordinary images of multiple-choice answer sheets into structured, reviewable grading results—reducing repetitive work while keeping uncertain cases visible to educators.

**FA**

> SmartSheet با استفاده از بینایی ماشین، تصاویر معمولی برگه‌های چندگزینه‌ای را به نتایج ساختاریافته و قابل بررسی تبدیل می‌کند؛ در نتیجه کار تکراری کاهش می‌یابد و موارد نامطمئن برای بررسی مدرس قابل مشاهده باقی می‌مانند.

## 3. Evidence wording / نحوه بیان شواهد

### Historical controlled result

**EN:** “The project documentation records a 95.2% average result on 10 controlled exam sheets.”

**FA:** «در مستندات پروژه، میانگین نتیجه ۹۵.۲٪ روی ۱۰ برگه آزمون کنترل‌شده ثبت شده است.»

### New stress test

**EN:** “A later six-scenario stress test covered 205 questions and recorded 147 correct results (71.7% weighted), revealing substantial weaknesses in some one-column and severe-perspective scenarios.”

**FA:** «یک تست استرس شش‌سناریویی بعدی شامل ۲۰۵ سؤال بود و ۱۴۷ پاسخ درست ثبت کرد (۷۱.۷٪ به‌صورت وزنی) و ضعف قابل توجه سیستم را در برخی حالت‌های تک‌ستونه و پرسپکتیو شدید نشان داد.»

## 4. What NOT to say / چه چیزهایی نگوییم

- “100% accurate in real-world use.”
- “Works on every smartphone.”
- “Production-ready for every exam sheet.”
- “Uses deep learning/CNN” unless that model is actually included in the final code.
- “95.2% real-world accuracy.”

## 5. Stronger story / داستان قوی‌تر پروژه

**Accessible assessment automation + transparent computer vision + human review + measurable engineering progress.**

**اتوماسیون قابل دسترس ارزیابی + بینایی ماشین شفاف + بررسی انسانی + پیشرفت مهندسی قابل اندازه‌گیری.**

## 6. Final pre-upload check / بررسی قبل از آپلود

- Project name is consistently **SmartSheet**.
- Final version number is consistent everywhere.
- No placeholder GitHub URL remains.
- The English documentation matches the actual ZIP.
- The benchmark claims match the evidence folder.
- The 95.2% and 71.7% results are clearly separated.
- The AI wording is technically honest.
- The demo video uses real application evidence where possible.
