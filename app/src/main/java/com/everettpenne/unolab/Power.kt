package com.everettpenne.unolab

import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.os.BatteryManager
import kotlin.math.abs

/**
 * The phone's draw from its battery, from the fuel gauge (no permission): current times voltage. Meaningful only on battery; on a charger the
 * gauge reads the charge going in. The figures are whole-phone, screen included, so energy for a task is its draw minus the idle draw before it.
 */
class Power(private val context: Context) {
    private val bm = context.getSystemService(BatteryManager::class.java)

    fun charging(): Boolean {
        val i = context.registerReceiver(null, IntentFilter(Intent.ACTION_BATTERY_CHANGED))
        return (i?.getIntExtra(BatteryManager.EXTRA_PLUGGED, 0) ?: 0) != 0
    }

    /** Watts now, or null when the gauge does not say. */
    fun watts(): Double? {
        val microAmps = bm.getLongProperty(BatteryManager.BATTERY_PROPERTY_CURRENT_NOW)
        if (microAmps == Long.MIN_VALUE || microAmps == 0L) return null
        val i = context.registerReceiver(null, IntentFilter(Intent.ACTION_BATTERY_CHANGED))
        val milliVolts = i?.getIntExtra(BatteryManager.EXTRA_VOLTAGE, 0) ?: 0
        if (milliVolts <= 0) return null
        return abs(microAmps) / 1e6 * (milliVolts / 1e3)
    }
}

/** Averages watts sampled while a task runs. */
class PowerMeter {
    private var sum = 0.0
    private var n = 0
    fun add(w: Double?) { if (w != null) { sum += w; n++ } }
    val mean: Double? get() = if (n == 0) null else sum / n
    val samples: Int get() = n
}

/** The sums, pure so they can be checked: energy for a task from its mean draw, the idle draw, and how long it ran. */
object Energy {
    fun joules(meanWatts: Double, idleWatts: Double, seconds: Double): Double = ((meanWatts - idleWatts).coerceAtLeast(0.0)) * seconds
    fun perUnit(joules: Double, units: Double): Double? = if (units <= 0) null else joules / units
}
