# Bunny Maze

## Direction

A playful, tactile toy maze. A cream bunny in a teal scarf explores sandy paths between raised slate-blue blocks. A blue home pad marks the start, a black-and-white checkered flag marks the finish, and red horned troublemakers mark the penalty cells. Shape and silhouette distinguish every game object as well as color.

The board is the main surface. A quiet ivory interface holds the level picker and seed controls; the top bar holds score, moves, and time. A permanent illustrated legend explains the four markers and the wall blocks. Controls have clear hover, selected, pressed, and disabled states.

## Tokens

- Background `#F5F4ED`; panels `#FFFEFA`; divider `#DDDCD3`.
- Text `#24364B`; secondary text `#5C6B77`; selected action `#BC4E26` with white text.
- Board surround `#D9E6E5`; floor `#F4ECD7` / `#EFE4CA`.
- Wall top `#71899E`; wall face `#526B82`; wall edge `#2E4258`.
- Player cream `#FFF7E8`, teal scarf `#197F85`, rose cheeks `#E9959B`.
- Start blue `#347BD2` with a white home symbol.
- Finish black-and-white checkered flag, gold base `#E9B74E`.
- Penalty red `#D93645`, dark outline `#692739`, ivory horns.
- Visited steps muted teal `#B9D8CA`.
- Headings bundled Fredoka; interface bundled Nunito. Type scale 12, 14, 16, 18, 24, 32 px.
- Corners: panels 18 px; controls 9 px; level buttons 7 px. Spacing 4, 8, 12, 16, 24 px.
- Authored pixel sprites use nearest-neighbor scaling. Walls have a lit top and a darker front edge.
- Motion: always-enabled 100 ms movement interpolation and short collision feedback, with immediate input. There is no motion toggle or keyboard shortcut to disable it.

## Camera and scale

The generated passage graph is compiled into a `(2n + 1)` board of equal-sized blocks. Both generated cells and the intermediate floor blocks in their corridors are real simulation positions. One key press moves exactly one block, the same pixel distance as one wall block's width. The bunny can stop on every floor square; it never jumps over a connector square. Wall blocks have no outgoing passages. Manual play, scoring, and headless observations all use this same block grid. Held or repeated key-down events do not trigger additional moves before key release. Small mazes fit the viewport. Large mazes start at a readable tile size and follow the bunny. `M` or **Fit map** reveals the whole maze; zoom buttons or the mouse wheel change scale. An overview shows the player, start, finish, and current camera bounds.
