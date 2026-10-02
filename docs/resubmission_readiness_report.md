# 🎯 MRICovNetX পেপার রিসাবমিশন রেডিনেস রিপোর্ট

**তারিখ:** ১ অক্টোবর ২০২৬  
**স্ট্যাটাস:** ⚠️ **~৯৫% রেডি — কিছু গুরুত্বপূর্ণ কাজ বাকি**

---

## ✅ এইমাত্র ফিক্স করলাম (৪টা ক্রিটিকাল বাগ)

| # | সমস্যা | স্ট্যাটাস |
|---|--------|----------|
| 1 | **Table 4 (CovBI-GRU Architecture) ভাঙা LaTeX** — সব Output Shape সেলে `$` সাইন মিসিং ছিল, Overleaf-এ compilation crash করতো | ✅ ফিক্সড |
| 2 | **Dataset-3 contradiction** — দুই জায়গায় (L464, L986) লেখা ছিল "reserved entirely for testing" অথচ D3 আসলে 5,712 train + 1,311 test এ split | ✅ ফিক্সড (দুই জায়গায়ই) |
| 3 | **D2-D4 patient-level split ব্যাখ্যা মিসিং** — Reviewer জিজ্ঞেস করবে কেন শুধু D1-এ patient split? | ✅ ফিক্সড — "Datasets 2–4 lack patient identifiers" বাক্য যোগ করলাম |
| 4 | **Table 8 (Baselines) থেকে Training Time মিসিং** — Reviewer R5.7 এটা চেয়েছিল, CSV-তে ডেটা ছিল কিন্তু table-এ ছিল না | ✅ ফিক্সড — column যোগ হয়ে গেছে |

---

## ✅ আগে থেকে সম্পূর্ণ (১৫টা আইটেম)

| বিষয় | ডিটেইল |
|-------|--------|
| টাইটেল আপডেট | "Multi-Format Dual-Branch Deep Neural Framework with Explainable AI" |
| অ্যাবস্ট্র্যাক্ট রিরাইট | Patient-level 82.73±0.51% রিপোর্ট করা হয়েছে |
| ইন্ট্রো মোটিভেশন প্যারা | ✅ R5.4 সমাধান |
| Related Works ৪ সাবসেকশন | CNN, Hybrid ML, Recurrent, Transformer ✅ R2.2 |
| ৬টি রিভিউয়ার-রিকোয়েস্টেড পেপার সাইট | ✅ R2.2, R3.8, R3.12 |
| CovBI-GRU Architecture Table (Table 4) | ✅ R3.6 (এইমাত্র syntax ফিক্সড) |
| Dataset-4 label mapping | ✅ R3.3, R3.11 |
| Overfitting clarification | ✅ R5.6 — validation loss checkpointing |
| হার্ডওয়্যার RTX 3060 | ✅ R3.1 — Tesla T4/Colab ১০০% রিমুভড |
| Deduplication analysis | ✅ Table + section |
| Patient-level split | ✅ Table + 82.73±0.51% + D2-D4 explanation |
| CovBI-GRU ablation (3-seed) | ✅ Table 7 — 5 variant × 3 seed |
| Modern baselines comparison | ✅ Table 8 — 6 architectures + training time |
| Quantitative XAI (EBPG/IoU) | ✅ Table 6 + Grad-CAM figure |
| Failure case analysis | ✅ Table + figure — 3 misclassified samples |

---

## ⚠️ বাকি আছে — সিদ্ধান্ত নিতে হবে

### 🔴 সমস্যা ১: CovNet22-এর D2/D3/D4 রেজাল্ট Single-Run (mean±SD নেই)

**কী সমস্যা:** Reviewer R3.5 সরাসরি বলেছে "mean ± SD" চাই **সব** ডেটাসেটে। কিন্তু এখন:
- Dataset-1 (CovBI-GRU): ✅ 82.73 ± 0.51% (3-seed)
- Dataset-2 (CovNet22): ❌ 96.17% (single run)
- Dataset-3 (CovNet22): ❌ 98.64% (single run)
- Dataset-4 (CovNet22): ❌ 96.92% (single run)

**ঝুঁকি:** মাঝারি-থেকে-উচ্চ। Reviewer দেখবে D1-এ mean±SD আছে কিন্তু D2-D4-তে নেই — inconsistency ধরবে।

**সমাধান:** CovNet22 মডেল 3 seed × 3 dataset = 9 বার ট্রেন করতে হবে  
**সময়:** তোমার RTX 3060-এ আনুমানিক **৩-৫ ঘণ্টা** (GPU চালু রাখলে)  
**আমি কি করতে পারি:** স্ক্রিপ্ট তৈরি করে দিতে পারি, তুমি রান করবে।

---

### 🟡 সমস্যা ২: ViT/DeiT Baseline মিসিং

**কী সমস্যা:** Reviewer R3.7 বলেছে "lightweight ViT/DeiT"-এর সাথে তুলনা করো। Table 8-তে শুধু CNN আছে, কোনো Transformer নেই।

**ঝুঁকি:** মাঝারি। Response letter-এ justify করা যায়, কিন্তু একটু experiment দিলে শক্তিশালী হতো।

**২টা অপশন:**
1. **রান করো** (ViT-Small বা Swin-Tiny, ~২ ঘণ্টা GPU) — আমি স্ক্রিপ্ট দেবো
2. **Response letter-এ justify করো** — "ViT variants require large-scale pretraining data and are 10-50× larger than CovNet22; our focus is lightweight clinical deployment"

---

### 🟡 সমস্যা ৩: ৩টা Unverified DOI

| পেপার | DOI | স্ট্যাটাস |
|--------|-----|----------|
| Shahzad (Tumor-Swin, Array) | `10.1016/j.array.2026.100954` | ⚠️ DOI resolve করে কিন্তু content verify করতে পারিনি |
| Leelavathi (Neuroscience Informatics) | `10.1016/j.neuri.2026.100283` | ❓ চেক করা হয়নি |
| Singh (Expert Systems) | `10.1111/exsy.13770` | ❓ চেক করা হয়নি |

**ঝুঁকি:** উচ্চ — যদি কোনোটা ভুল হয়, পুরো পেপারের বিশ্বাসযোগ্যতা ক্ষতিগ্রস্ত হবে।

**তোমার কাজ:** ব্রাউজারে গিয়ে `https://doi.org/DOI_NUMBER` টাইপ করে প্রতিটা DOI চেক করো। যেটা মিলবে না — বাদ দাও।

---

### 🟢 সমস্যা ৪: LaTeX Compilation Test

**কী সমস্যা:** পেপার কখনো compile করে দেখা হয়নি। Overleaf-এ আপলোড করলে error আসতে পারে।

**সমাধান:** `overleaf_revised.zip` (বা আপডেটেড ভার্সন) Overleaf-এ আপলোড করো → compile করো → error দেখলে আমাকে বলো।

**সময়:** ১০-১৫ মিনিট

---

## 📊 সারসংক্ষেপ: কতটুকু রেডি?

```
███████████████████░░  ~৯৫% সম্পূর্ণ
```

| ক্যাটাগরি | স্ট্যাটাস |
|-----------|----------|
| ম্যানুস্ক্রিপ্ট টেক্সট | ✅ ৯৮% সম্পূর্ণ |
| টেবিল ও ফিগার | ✅ সব আছে |
| Bibliography | ✅ ৫৮ citations, zero missing keys |
| Response to Reviewers | ✅ সম্পূর্ণ |
| Experimental rigor (D1) | ✅ 3-seed, patient-split |
| Experimental rigor (D2-D4) | ⚠️ Single-run (mean±SD নেই) |
| ViT/DeiT baseline | ⚠️ মিসিং |
| DOI verification | ⚠️ ৩/৬ unverified |
| LaTeX compilation | ❓ করা হয়নি |

---

## 🎯 আমার রেকমেন্ডেশন

### **এখনই সাবমিট করতে চাইলে:**
পেপার "acceptable" অবস্থায় আছে — major revision-এর ৯০%+ কাজ হয়ে গেছে। কিন্তু **re-rejection-এর ঝুঁকি আছে** D2-D4-এ mean±SD না থাকলে।

### **পারফেক্ট করতে চাইলে (আমার সাজেশন):**

| ধাপ | কাজ | সময় | কে করবে |
|-----|-----|------|---------|
| 1️⃣ | ৩টা DOI ব্রাউজারে ভেরিফাই | ১৫ মিনিট | **তুমি** |
| 2️⃣ | CovNet22 3-seed রান (D2, D3, D4) | ৩-৫ ঘণ্টা (GPU) | **তুমি** (স্ক্রিপ্ট আমি দেবো) |
| 3️⃣ | ViT-Small baseline রান | ১-২ ঘণ্টা (GPU) | **তুমি** (স্ক্রিপ্ট আমি দেবো) |
| 4️⃣ | নতুন রেজাল্ট LaTeX-এ আপডেট | ৩০ মিনিট | **আমি** |
| 5️⃣ | নতুন zip + Overleaf compile test | ১৫ মিনিট | **তুমি** |

**মোট সময়:** ~৬-৮ ঘণ্টা (বেশিরভাগ GPU wait time)

> [!TIP]
> তুমি বলেছিলে "time laguk kora suru koro... perfact banaw" — তাই আমার সাজেশন হলো ধাপ 1-5 করো, তারপর সাবমিট দাও। এতে re-rejection-এর ঝুঁকি অনেক কমবে।
