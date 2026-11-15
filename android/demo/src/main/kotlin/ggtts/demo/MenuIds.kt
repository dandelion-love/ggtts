// The ⋮ menu's item ids, in one place. Two items given one id open each other's action; MenuIdsTest holds them
// distinct and holds MainActivity to these names rather than numbers.
package ggtts.demo

object MenuIds {
    const val IMPORT = 1
    const val ABOUT = 2
    const val ADVANCED = 3
    const val MODELS_FOLDER = 4
    // 5 is the Licences sheet's in its own pending change; a later merge adds it here, and the test then checks it.
    const val SAVED_READS = 6

    /** Every id the menu uses. */
    val ALL = listOf(IMPORT, ABOUT, ADVANCED, MODELS_FOLDER, SAVED_READS)
}
