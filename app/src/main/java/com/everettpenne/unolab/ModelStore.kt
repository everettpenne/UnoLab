package com.everettpenne.unolab

import android.content.Context
import android.net.Uri
import java.io.File

/**
 * Where the model lives. It arrives in the app's folder on shared storage, which `adb push` can write to with no permission
 * (/sdcard/Android/data/com.everettpenne.unolab/files/), or from a file picked with the system picker; it is then moved into the app's
 * internal storage and loaded from there. Mapped from shared storage, every page of it came in through Android's FUSE layer: on the emulator
 * the model loaded at about 6 MB/s, minutes for 1.3 GB.
 */
class ModelStore(private val context: Context) {
    /** Where models are dropped. */
    val dir: File get() = requireNotNull(context.getExternalFilesDir(null))
    /** Where they are loaded from. */
    private val home: File get() = File(context.filesDir, "models").apply { mkdirs() }

    private fun ggufs(d: File) = d.listFiles { f -> f.isFile && f.name.endsWith(".gguf") }?.toList().orEmpty()

    /** A model dropped in [dir] and not yet moved in, if any. */
    fun waiting(): File? = ggufs(dir).maxByOrNull { it.lastModified() }

    /** The newest model moved in. */
    fun find(): File? = ggufs(home).maxByOrNull { it.lastModified() }

    /** Moves [from] into internal storage (a copy, then the original removed), reporting the fraction done. */
    fun moveIn(from: File, progress: (Float) -> Unit): File {
        val part = File(home, from.name + ".part")
        val size = from.length()
        from.inputStream().use { input -> part.outputStream().use { out -> copy(input, out, size, progress) } }
        val target = File(home, from.name)
        ggufs(home).forEach { it.delete() }   // one model at a time: they are over a gigabyte each
        check(part.renameTo(target)) { "The model could not be moved in" }
        from.delete()
        return target
    }

    private fun copy(input: java.io.InputStream, out: java.io.OutputStream, size: Long, progress: (Float) -> Unit) {
        val buf = ByteArray(1 shl 20)
        var done = 0L
        while (true) {
            val n = input.read(buf); if (n < 0) break
            out.write(buf, 0, n); done += n
            if (size > 0) progress((done.toDouble() / size).toFloat())
        }
    }

    /** Copies [uri] into the folder, reporting the fraction done. */
    fun import(uri: Uri, progress: (Float) -> Unit): File {
        val name = context.contentResolver.query(uri, arrayOf(android.provider.OpenableColumns.DISPLAY_NAME), null, null, null)?.use { c ->
            if (c.moveToFirst()) c.getString(0) else null
        }?.takeIf { it.endsWith(".gguf") } ?: "model.gguf"
        val size = context.contentResolver.openAssetFileDescriptor(uri, "r")?.use { it.length } ?: -1L
        val part = File(home, "$name.part")
        context.contentResolver.openInputStream(uri).use { input ->
            requireNotNull(input) { "The file could not be opened" }
            part.outputStream().use { out -> copy(input, out, size, progress) }
        }
        val target = File(home, name)
        ggufs(home).forEach { it.delete() }
        check(part.renameTo(target)) { "The copy could not be finished" }
        return target
    }
}
