# ⚡ Deploy në Railway.app - 5 Minuta

## Më i Shpejti Opsion për Hosting Falas!

### ✅ Përgatitja (1 herë)

**1. Krijo GitHub Repository**
```powershell
cd SchedulingAPI
git init
git add .
git commit -m "Deploy to Railway"
```

Pastaj shko te GitHub.com:
1. Kliko "New repository" 
2. Emërto: `TV-Scheduling-API`
3. Mos zgjedh asgjë tjetër (no README, no .gitignore)
4. Kliko "Create repository"

**2. Push Code**
```powershell
git remote add origin https://github.com/YOUR-USERNAME/TV-Scheduling-API.git
git branch -M main
git push -u origin main
```

---

### 🚀 Deploy në Railway (5 Minuta)

**Hapi 1: Krijo Llogari**
1. Shko te: https://railway.app
2. Kliko "Login with GitHub"
3. Authorize Railway

**Hapi 2: Krijo Projekt të Ri**
1. Kliko "New Project"
2. Zgjedh "Deploy from GitHub repo"
3. Zgjedh repository: `TV-Scheduling-API`
4. Kliko projektin

**Hapi 3: Configure (Automatic)**
Railway do ta detect-ojë automatikisht:
- ✅ .NET 9.0 project
- ✅ Dockerfile (që e krijuam)
- ✅ Port configuration

Shtypni "Deploy" dhe prit 2-3 minuta.

**Hapi 4: Merr Public URL**
1. Kur deployment përfundon, kliko "Settings"
2. Scroll poshtë te "Networking"
3. Kliko "Generate Domain"
4. Do të marrësh URL si: `https://tv-scheduling-api-production.up.railway.app`

**Hapi 5: Testo API-në**
Hap browser:
```
https://YOUR-URL.up.railway.app/swagger
```

---

## 🎉 Gata!

API-ja tani është live dhe e aksesueshme nga gjithë bota!

### URLs:
- **Swagger UI**: `https://your-url.up.railway.app/swagger`
- **API Endpoint**: `https://your-url.up.railway.app/api/schedule`

### Test me cURL:
```bash
curl -X POST https://your-url.up.railway.app/api/schedule \
  -H "Content-Type: application/json" \
  -d @example_input.json
```

---

## 💰 Kosto

Railway Free Tier:
- ✅ **$5 falas credit/muaj**
- ✅ Mjafton për projekte të vogla
- ✅ S'kërkon kartë krediti për të filluar
- ⚠️ Pas $5, do të duhet të upgradosh ose API ndalon (por për testim, mjafton!)

---

## 🔄 Si të Bësh Update

Kur të ndryshosh kodin:
```powershell
git add .
git commit -m "Updated algorithm"
git push
```

Railway do të re-deploy automatikisht! 🚀

---

## 🆘 Probleme?

**Problem: Build failed**
- Shiko logs në Railway dashboard
- Sigurohu që Dockerfile është në root të projektit

**Problem: App crashes**
- Shiko "Deploy Logs" në Railway
- Verifikoni që porti është 8080 (ashtu siç e konfiguruat)

**Problem: S'hapet Swagger**
- Sigurohu që po shkon te `/swagger` (jo vetëm root URL)
- Prit disa sekonda pas deployment

---

## ✨ Bonus: Environment Variables

Nëse ke nevojë për secrets (psh database connection):
1. Shko te Railway dashboard
2. Kliko "Variables" tab
3. Shto environment variables
4. Re-deploy

---

Gëzuar deployment! 🎊

