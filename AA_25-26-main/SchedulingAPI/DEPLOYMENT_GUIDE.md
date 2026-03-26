# 🚀 Udhëzues për Deployment (Hosting Falas)

## Opsioni 1: Railway.app (MË I LEHTË - E REKOMANDUAR!)

### Hapi 1: Krijo Llogari në Railway
1. Shko te https://railway.app
2. Kliko "Start a New Project"
3. Login me GitHub account

### Hapi 2: Deploy Projektin
1. Kliko "New Project"
2. Zgjedh "Deploy from GitHub repo"
3. Lidh GitHub account-in tënd
4. Zgjedh repository ku e ke projektin
5. Railway do ta detect-ojë automatikisht që është .NET projekt
6. Kliko "Deploy"

### Hapi 3: Merr URL-në
- Pas deployment, Railway të jep një URL si: `https://schedulingapi-production.up.railway.app`
- API-ja do jetë e aksesueshme në: `https://your-url.railway.app/swagger`

### Kosto: 
- ✅ **100% FALAS** për 500 orë/muaj (mjafton për testim)

---

## Opsioni 2: Render.com

### Hapi 1: Krijo Llogari
1. Shko te https://render.com
2. Regjistrohu me GitHub

### Hapi 2: Deploy
1. Kliko "New +" → "Web Service"
2. Lidh GitHub repository
3. Përdor këto settings:
   - **Build Command**: `dotnet publish -c Release -o out`
   - **Start Command**: `dotnet out/SchedulingAPI.dll`
   - **Environment**: Docker
4. Kliko "Create Web Service"

### Hapi 3: Merr URL-në
- Do të marrësh URL si: `https://schedulingapi.onrender.com`
- Swagger: `https://schedulingapi.onrender.com/swagger`

### Kosto:
- ✅ **FALAS** tier available
- ⚠️ Mund të "fjejë" pas inaktiviteti (merr disa sekonda të aktivizohet përsëri)

---

## Opsioni 3: Azure App Service (Free Tier)

### Hapi 1: Krijo Llogari Azure
1. Shko te https://azure.microsoft.com/free
2. Regjistrohu (kërkon kartë krediti, por NUK të ngarkon për free tier)

### Hapi 2: Deploy nga Visual Studio/VS Code
1. Hap projektin në Visual Studio
2. Right-click në projekt → "Publish"
3. Zgjedh "Azure" → "Azure App Service (Windows/Linux)"
4. Zgjedh "Free" tier (F1)
5. Kliko "Publish"

### Hapi 3: Merr URL-në
- Do të marrësh URL si: `https://schedulingapi.azurewebsites.net`
- Swagger: `https://schedulingapi.azurewebsites.net/swagger`

### Kosto:
- ✅ **FALAS** F1 tier (1GB RAM, 60 min/day compute)

---

## Opsioni 4: Fly.io

### Hapi 1: Instalo Fly CLI
```powershell
iwr https://fly.io/install.ps1 -useb | iex
```

### Hapi 2: Login dhe Deploy
```powershell
fly auth login
cd SchedulingAPI
fly launch
```

Follow prompts, zgjedh region dhe emrin.

### Hapi 3: Deploy
```powershell
fly deploy
```

### Kosto:
- ✅ **FALAS** tier (3 shared-cpu-1x VMs)

---

## 📋 Përmbledhje - Cila të Zgjedhësh?

| Platform | Vështirësi | Falas | Rekomandim |
|----------|------------|-------|------------|
| **Railway.app** | ⭐ Shumë e lehtë | ✅ Po | 🏆 **BEST për fillestare** |
| **Render.com** | ⭐⭐ E lehtë | ✅ Po | ✅ E mirë alternativë |
| **Azure** | ⭐⭐⭐ Mesatare | ✅ Po* | ✅ Profesionale |
| **Fly.io** | ⭐⭐⭐ Mesatare | ✅ Po | ✅ E mirë |

*Azure kërkon kartë krediti për verifikim, por nuk ngarkon për free tier

---

## 🎯 Rekomandimi Im: Railway.app

Për ty, **Railway.app** është më e mira sepse:
1. ✅ Deployment me një klik
2. ✅ S'kërkon kartë krediti
3. ✅ Auto-detect .NET projects
4. ✅ Jep URL publike menjëherë
5. ✅ Logs të qarta
6. ✅ 500 orë falas/muaj (mjafton për testim)

---

## 🚨 Nëse Nuk Ke GitHub Repository

Nëse projektin e ke vetëm lokalisht, bëj këto:

### 1. Krijo Repository në GitHub
1. Shko te https://github.com
2. Kliko "New repository"
3. Emërto projektin (psh: "TV-Scheduling-API")
4. Kliko "Create repository"

### 2. Push Projektin
```powershell
cd SchedulingAPI
git init
git add .
git commit -m "Initial commit"
git branch -M main
git remote add origin https://github.com/USERNAME/TV-Scheduling-API.git
git push -u origin main
```

### 3. Pastaj Deploy në Railway ose Render
- Lidh GitHub repository me Railway/Render
- Deploy automatikisht

---

## 📞 Ndihmë?

Nëse ke vështirësi, më thuaj në cilin platform do të provosh dhe të ndihmoj step-by-step! 🚀

## ✨ Pas Deployment

Kur të jetë live, API-ja do jetë e aksesueshme nga kudo:
- Swagger: `https://your-url.com/swagger`
- API Endpoint: `https://your-url.com/api/schedule`

Mund ta testosh nga telefoni, laptopi, kudo! 🌍

