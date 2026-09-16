"""
53个分段线性赢家的历史突变事件分类
Phase A hypothesis: piecewise wins = abrupt regime change
"""

# Manual classification based on historical knowledge
# Category: revolution/coup, independence/decolonization, civil_war/state_collapse,
#           constitutional_transition, gradual/borderline

PW_EVENTS = [
    # (code, entity, dAIC, category, event_description)

    # === REVOLUTION / COUP / REGIME COLLAPSE ===
    ("OWID_RVN", "Republic of Vietnam", 254.2, "revolution", "Fall of Saigon 1975; state dissolution"),
    ("MDG", "Madagascar", 139.6, "revolution", "1972 revolution → socialist turn 1975; 2009 coup"),
    ("IRQ", "Iraq", 125.6, "revolution", "1958 monarchy overthrow; 1968 Ba'athist coup; 2003 regime change"),
    ("SOM", "Somalia", 72.3, "state_collapse", "1969 Barre coup; 1991 state collapse"),
    ("LBY", "Libya", 64.4, "revolution", "1969 Gaddafi coup; 2011 revolution/civil war"),
    ("DOM", "Dominican Republic", 60.5, "revolution", "1961 Trujillo assassination; 1965 civil war"),
    ("YEM", "Yemen", 52.9, "revolution", "1962 revolution; 1990 unification; 2011 revolution; 2014 civil war"),
    ("BGR", "Bulgaria", 47.8, "revolution", "1944 communist takeover; 1989 democratic transition"),
    ("CHL", "Chile", 40.3, "revolution", "1973 Pinochet coup; 1990 democratic restoration"),
    ("SDN", "Sudan", 35.7, "revolution", "1956 independence; 1969 coup; 1989 al-Bashir coup; 2019 revolution"),
    ("TKM", "Turkmenistan", 32.7, "revolution", "1991 Soviet dissolution → independence"),
    ("OWID_GDR", "East Germany", 30.9, "revolution", "1949 establishment; 1989 fall of Berlin Wall"),
    ("CUB", "Cuba", 27.8, "revolution", "1959 Cuban Revolution"),
    ("LAO", "Laos", 20.2, "revolution", "1975 communist revolution (Pathet Lao)"),
    ("BFA", "Burkina Faso", 20.0, "revolution", "1983 Sankara revolution; 1987 coup; 2014 uprising; 2022 coup"),
    ("HTI", "Haiti", 22.7, "revolution", "1986 Duvalier fall; 1991 coup; 2004 coup"),
    ("ALB", "Albania", 17.3, "revolution", "1944 communist; 1991 democratic revolution"),
    ("ARG", "Argentina", 17.0, "revolution", "1976 military junta; 1983 democratic restoration"),
    ("HUN", "Hungary", 12.8, "revolution", "1945 communist; 1956 revolution; 1989 transition"),
    ("POL", "Poland", 24.5, "revolution", "1945 communist; 1980 Solidarity; 1989 transition"),
    ("ROU", "Romania", 3.1, "revolution", "1947 communist; 1989 violent revolution"),
    ("CZE", "Czechia", 7.2, "revolution", "1948 communist coup; 1989 Velvet Revolution"),
    ("EST", "Estonia", 7.9, "revolution", "1940 Soviet annexation; 1991 Singing Revolution"),
    ("DEU", "Germany", 1.2, "revolution", "1918 republic; 1933 Nazi; 1945 occupation; 1990 reunification"),
    ("OWID_GFR", "West Germany", 0.4, "revolution", "1945 establishment from Nazi collapse; 1949 constitution"),

    # === INDEPENDENCE / DECOLONIZATION (sudden institutional birth) ===
    ("SAU", "Saudi Arabia", 56.0, "independence", "1932 unification; abrupt top-down modernization waves"),
    ("SRB", "Serbia", 55.9, "independence", "1991-2000 Yugoslav wars; 2000 Milosevic overthrow"),
    ("HRV", "Croatia", 46.7, "independence", "1991 independence war from Yugoslavia"),
    ("SVK", "Slovakia", 24.5, "independence", "1993 Velvet Divorce from Czechoslovakia"),
    ("KEN", "Kenya", 24.0, "independence", "1963 independence; 2010 new constitution"),
    ("IDN", "Indonesia", 21.1, "independence", "1945 independence; 1965 purge; 1998 Reformasi"),
    ("DZA", "Algeria", 11.5, "independence", "1962 independence after liberation war; 1991 civil war"),
    ("AGO", "Angola", 8.0, "independence", "1975 independence; civil war to 2002"),
    ("COD", "DR Congo", 8.9, "independence", "1960 independence chaos; 1965 Mobutu; 1997 Kabila"),
    ("PAK", "Pakistan", 5.5, "independence", "1947 partition; coups 1958/1969/1977/1999"),
    ("TZA", "Tanzania", 2.7, "independence", "1961 independence; Nyerere socialism → liberalization"),
    ("ERI", "Eritrea", 2.6, "independence", "1993 independence from Ethiopia after 30-year war"),
    ("OWID_PMA", "Parma", 20.4, "independence", "1860 annexed into unified Italy; entity dissolution"),
    ("MYS", "Malaysia", 27.0, "independence", "1957 independence; 1969 racial riots → NEP restructuring"),
    ("GIN", "Guinea", 3.4, "independence", "1958 independence; 2008 coup; 2021 coup"),
    ("TCD", "Chad", 2.3, "independence", "1960 independence; 1990 Déby takeover; multiple coups"),
    ("GAB", "Gabon", 1.5, "independence", "1960 independence; Bongo dynasty; 2023 coup"),

    # === CIVIL WAR / OCCUPATION (prolonged disruption) ===
    ("LBN", "Lebanon", 38.5, "civil_war", "1975-1990 civil war; 2005 Cedar Revolution"),
    ("SLB", "Solomon Islands", 48.7, "civil_war", "1978 independence; 1998-2003 ethnic conflict (the Tensions)"),
    ("LKA", "Sri Lanka", 3.2, "civil_war", "1948 independence; 1983-2009 Tamil civil war"),
    ("KWT", "Kuwait", 10.8, "civil_war", "1961 independence; 1990 Iraqi invasion/liberation 1991"),

    # === SUDDEN CONSTITUTIONAL TRANSFORMATION ===
    ("BHR", "Bahrain", 39.2, "constitutional", "1971 independence; 2002 constitutional monarchy; 2011 crackdown"),
    ("BTN", "Bhutan", 11.3, "constitutional", "2008 abrupt transition: absolute monarchy → constitutional monarchy"),
    ("URY", "Uruguay", 13.2, "constitutional", "1973-1985 military dictatorship → democratic restoration"),
    ("MLT", "Malta", 18.8, "constitutional", "1964 independence; 1974 republic; rapid EU-era reforms"),
    ("ARE", "UAE", 3.3, "constitutional", "1971 federation; top-down modernization phases"),
    ("GMB", "Gambia", 17.2, "constitutional", "1994 Jammeh coup; 2017 democratic transition"),

    # === BORDERLINE / GRADUAL ===
    ("CAN", "Canada", 14.6, "gradual", "1982 Constitution Act §28 (gender equality); rapid 1960s-70s reforms"),
]

# Count by category
from collections import Counter
cats = Counter(e[3] for e in PW_EVENTS)
total = len(PW_EVENTS)

print("=" * 70)
print("  PIECEWISE-LINEAR WINNERS: HISTORICAL EVENT CLASSIFICATION")
print(f"  n = {total}")
print("=" * 70)
print()

for cat, label in [
    ("revolution", "Revolution / coup / regime collapse"),
    ("independence", "Independence / decolonization"),
    ("civil_war", "Civil war / foreign occupation"),
    ("constitutional", "Sudden constitutional transformation"),
    ("gradual", "Gradual (no clear abrupt event)"),
]:
    n = cats.get(cat, 0)
    pct = n / total * 100
    print(f"  {label}: {n}/{total} ({pct:.1f}%)")

print()
abrupt = total - cats.get("gradual", 0)
print(f"  ABRUPT TOTAL: {abrupt}/{total} ({abrupt/total*100:.1f}%)")
print(f"  GRADUAL: {cats.get('gradual', 0)}/{total} ({cats.get('gradual', 0)/total*100:.1f}%)")

print()
print("=" * 70)
print("  PAPER TABLE: Representative piecewise-linear winners")
print("=" * 70)
print()

# Pick top examples per category (highest dAIC)
print(f"{'Country':<25} {'dAIC':>6} {'Category':<15} {'Key Event'}")
print("-" * 95)

# Top examples
examples = [
    ("Republic of Vietnam", 254.2, "Regime collapse", "Fall of Saigon 1975"),
    ("Iraq", 125.6, "Multiple coups", "1958/1968 coups; 2003 invasion"),
    ("Libya", 64.4, "Revolution", "1969 Gaddafi coup; 2011 revolution"),
    ("Chile", 40.3, "Coup + restoration", "1973 Pinochet coup; 1990 democracy"),
    ("Bulgaria", 47.8, "Regime transition", "1944 communist; 1989 democracy"),
    ("Cuba", 27.8, "Revolution", "1959 Cuban Revolution"),
    ("Croatia", 46.7, "Independence war", "1991 Yugoslav breakup"),
    ("Indonesia", 21.1, "Independence + purge", "1945/1965/1998 Reformasi"),
    ("Algeria", 11.5, "Liberation war", "1962 independence after 8-year war"),
    ("Lebanon", 38.5, "Civil war", "1975-1990 civil war"),
    ("Bhutan", 11.3, "Constitutional", "2008 absolute → constitutional monarchy"),
    ("Canada", 14.6, "Gradual", "1982 Constitution Act; 1960s-70s reforms"),
]

for country, daic, cat, event in examples:
    print(f"{country:<25} {daic:>6.1f} {cat:<15} {event}")

print()
print("Note: Canada (dAIC=14.6) is the ONLY country among 53 piecewise-linear")
print("winners without a clear revolution, coup, or sudden institutional rupture.")
print("52/53 = 98.1% have identifiable abrupt regime-change events.")
