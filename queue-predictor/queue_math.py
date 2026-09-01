"""
queue_math.py
-------------
Queueing theory calculations for M/M/1 queue models.

M/M/1 Queue Assumptions:
  - Single server (one counter)
  - Poisson arrivals with rate λ (lambda) customers per minute
  - Exponential service times with rate μ (mu) customers per minute
  - FIFO discipline, infinite queue capacity, infinite population

Key formulas used (for judge explanation):
  - Traffic intensity:  ρ = λ / μ   (utilisation; must be < 1 for stability)
  - Mean queue length:  Lq = ρ² / (1 − ρ)
  - Wait in queue:      Wq = Lq / λ  =  ρ / (μ − λ)  =  λ / (μ(μ − λ))
  - Simplified (empirical approx when current count is known):
                        Wq ≈ N / μ   (N = people currently in queue)
"""


def estimate_wait_time(people_in_queue: int, service_rate_per_min: float) -> float | None:
    """
    Simple empirical wait-time estimate: Wq ≈ N / μ.

    This approximation is intuitive: if there are N people waiting and the
    counter serves μ people per minute, the last person waits roughly N/μ
    minutes. Useful when we know the snapshot count but not the live arrival
    rate.

    Formula:
        Wq = people_in_queue / service_rate_per_min   (minutes)

    Args:
        people_in_queue:      Number of customers currently in the queue (N).
        service_rate_per_min: Counter throughput μ in customers per minute.

    Returns:
        Estimated wait time in minutes, or None if service_rate ≤ 0.
    """
    if service_rate_per_min <= 0:
        return None   # undefined — avoid division by zero
    return round(people_in_queue / service_rate_per_min, 1)


def estimate_wait_time_mm1(arrival_rate: float, service_rate: float) -> dict:
    """
    Full M/M/1 queue formula: Wq = λ / (μ(μ − λ)).

    Only valid when the system is STABLE, i.e., arrival_rate < service_rate
    (equivalently, traffic intensity ρ = λ/μ < 1).  When unstable, queues
    grow without bound — flagged as "overloaded".

    Formulas:
        ρ  = λ / μ                        (traffic intensity / utilisation)
        Lq = ρ² / (1 − ρ)                 (mean number waiting in queue)
        Wq = Lq / λ  =  λ / (μ(μ − λ))   (mean wait time in queue, minutes)

    Args:
        arrival_rate:  λ — mean customer arrivals per minute.
        service_rate:  μ — mean customers served per minute.

    Returns:
        Dict with keys:
            stable       (bool)   — whether the system is stable
            rho          (float)  — traffic intensity ρ
            Lq           (float)  — mean queue length (customers waiting)
            Wq_minutes   (float)  — mean wait time in minutes (None if unstable)
            status       (str)    — human-readable status
    """
    result = {"stable": False, "rho": None, "Lq": None, "Wq_minutes": None, "status": ""}

    if service_rate <= 0:
        result["status"] = "Invalid: service_rate must be > 0"
        return result

    rho = arrival_rate / service_rate
    result["rho"] = round(rho, 4)

    if arrival_rate >= service_rate:
        result["stable"] = False
        result["status"] = (
            f"OVERLOADED / UNSTABLE -- rho={rho:.2f} >= 1. "
            f"Queue grows without bound. Add capacity immediately."
        )
        return result

    Lq = (rho ** 2) / (1 - rho)
    Wq = arrival_rate / (service_rate * (service_rate - arrival_rate))

    result.update({
        "stable":     True,
        "Lq":         round(Lq, 2),
        "Wq_minutes": round(Wq, 2),
        "status":     f"Stable -- rho={rho:.2f}, avg wait={Wq:.1f} min, avg queue={Lq:.1f} people",
    })
    return result


# ──────────────────────────────────────────────
# Quick sanity test
# ──────────────────────────────────────────────
if __name__ == "__main__":
    print("=== Queue Math Sanity Test ===\n")

    # Test estimate_wait_time
    test_cases = [(18, 4.0), (4, 3.5), (21, 2.5), (0, 3.0), (10, 0)]
    print("Simple Wq = N/mu:")
    for n, mu in test_cases:
        wt = estimate_wait_time(n, mu)
        print(f"  N={n:3d}, mu={mu} -> Wq = {wt} min")

    print("\nFull M/M/1  Wq = lam/(mu*(mu-lam)):")
    mm1_cases = [
        (3.0, 4.0, "Bank Savings (stable)"),
        (5.0, 4.5, "Hospital OPD (unstable)"),
        (2.0, 2.5, "College Docs (stable)"),
        (6.0, 5.5, "Railway Booking (unstable)"),
    ]
    for lam, mu, label in mm1_cases:
        r = estimate_wait_time_mm1(lam, mu)
        print(f"  [{label}] lam={lam}, mu={mu} -> {r['status']}")
