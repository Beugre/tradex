initial = 1000.0
final   = 5051.27
total   = final - initial
years   = 6

cagr         = (final / initial) ** (1 / years) - 1
monthly_rate = (1 + cagr) ** (1/12) - 1
daily_rate   = (1 + cagr) ** (1/365.25) - 1

avg_yr  = total / years
avg_mo  = total / (years * 12)
avg_day = total / (years * 365.25)

y_pnl   = [1012.88, 110.58, -31.64, 486.00, 87.31, 189.41]
labels  = ["Y1 2020-21", "Y2 2021-22", "Y3 2022-23 PERTE", "Y4 2023-24", "Y5 2024-25", "Y6 2025-26"]

W = 72
print(f"\n{'='*W}")
print("  BULL PUR x2  —  $1 000  ->  $5 051  sur 6 ans")
print(f"{'='*W}")
print(f"\n  CAGR reel annualise :  {cagr*100:+.2f}%/an")
print(f"  Taux mensuel        :  {monthly_rate*100:+.2f}%/mois")
print(f"  Taux journalier     :  {daily_rate*100:+.3f}%/jour")

print(f"\n{'-'*W}")
print("  EN % du capital en cours (rate geometrique)")
print(f"{'-'*W}")
print(f"  {'Periode':<10}  {'% / jour':>10}  {'% / mois':>10}  {'% / an':>10}")
print(f"  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*10}")
print(f"  {'Toujours':<10}  {daily_rate*100:>+9.3f}%  {monthly_rate*100:>+9.2f}%  {cagr*100:>+9.2f}%")

print(f"\n{'-'*W}")
print("  EN DOLLARS — capital initial $1 000 (debut annee 1)")
print(f"{'-'*W}")
print(f"  Par jour    :  ${initial * daily_rate:>7.2f}/jour")
print(f"  Par mois    :  ${initial * monthly_rate:>7.2f}/mois")
print(f"  Par an      :  ${initial * cagr:>7.2f}/an")

print(f"\n{'-'*W}")
print("  EN DOLLARS — capital final $5 051 (fin annee 6)")
print(f"{'-'*W}")
print(f"  Par jour    :  ${final * daily_rate:>7.2f}/jour")
print(f"  Par mois    :  ${final * monthly_rate:>7.2f}/mois")
print(f"  Par an      :  ${final * cagr:>7.2f}/an")

print(f"\n{'-'*W}")
print("  MOYENNE SIMPLE  (total PnL / duree, reflete le capital croissant)")
print(f"{'-'*W}")
print(f"  Par jour    :  ${avg_day:>7.2f}/jour")
print(f"  Par mois    :  ${avg_mo:>7.2f}/mois")
print(f"  Par an      :  ${avg_yr:>7.2f}/an")

print(f"\n{'-'*W}")
print("  DETAIL PAR ANNEE   (depuis $1 000 independant chaque annee)")
print(f"{'-'*W}")
print(f"  {'Annee':<20}  {'PnL/an':>9}  {'PnL/mois':>9}  {'PnL/jour':>9}")
print(f"  {'-'*20}  {'-'*9}  {'-'*9}  {'-'*9}")
for lbl, pnl in zip(labels, y_pnl):
    print(f"  {lbl:<20}  ${pnl:>+8.2f}  ${pnl/12:>+8.2f}  ${pnl/365.25:>+8.3f}")
moy = sum(y_pnl)/6
print(f"  {'Moyenne 6 ans':<20}  ${moy:>+8.2f}  ${moy/12:>+8.2f}  ${moy/365.25:>+8.3f}")
print(f"{'='*W}\n")
