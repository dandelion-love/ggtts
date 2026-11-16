// The demo's words on a phone: its string resources, in the language the phone (or the app's own locale setting) asks
// for, with English where a translation is missing. The plain-Kotlin classes take it as their [Words].
package ggtts.demo

import android.content.Context

class ResourceWords(context: Context) : Words {
    private val app = context.applicationContext
    private val ids = HashMap<String, Int>()

    override fun get(key: String, vararg args: Any?): String {
        val id = synchronized(ids) {
            ids.getOrPut(key) { app.resources.getIdentifier(key, "string", app.packageName) }
        }
        // A name the resources lack is a programming error; the English text is still better than an empty label.
        if (id == 0) return EnglishWords.get(key, *args)
        return if (args.isEmpty()) app.getString(id) else app.getString(id, *args)
    }
}
