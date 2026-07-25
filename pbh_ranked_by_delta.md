# PBH prompts ranked by Δ = prompt_sim − random

跨类别的真实 mechanism
把三类合并看，prompt_sim 的真实优势区是:

✅ 大、独特、轮廓清晰 的物体（robots, marbles, balloons, cars, dogs, children, large bowls）
✅ 干净的背景 (sky, lawn, velvet, ceramic bowl) 而非室内杂物
✅ 3+ 个 semantically distinct 的对象 (而非同类靠颜色区分)
✅ 高对比配色

真实劣势区:

❌ 杂乱/自然/室内背景 (forest, desk, kitchen counter)
❌ 小物体 (buttons, pencils, ladybugs)
❌ 同类只靠属性区分 (tulip vs tulip, book vs books)
❌ 遮挡/水中 (branches, fishbowl)

Combined from `prompts.jsonl` (full) + `prompts_strong.jsonl` (strong).  
Δ averaged across seeds × providers × samples × questions.  
`avg_Δ` = (Δ_L + Δ_XL) / 2.


## counting_hard (145 prompts)

Distribution of avg_Δ:

  - -1.00 ≤ Δ < -0.10: 14
  - -0.10 ≤ Δ < -0.05: 15
  - -0.05 ≤ Δ < +0.00: 30
  - +0.00 ≤ Δ < +0.05: 54
  - +0.05 ≤ Δ < +0.10: 20
  - +0.10 ≤ Δ < +1.00: 12

| ds | id | n_q | Δ_L | Δ_XL | avg_Δ | prompt |
|---|---|---|---|---|---|---|
| full | pbh-counting_hard-021 | 1 | +0.417 | +0.083 | **+0.250** | Four identical robots standing in a row. |
| full | pbh-counting_hard-057 | 1 | +0.083 | +0.250 | **+0.167** | Four windmills on a grassy hillside. |
| full | pbh-counting_hard-047 | 1 | +0.000 | +0.292 | **+0.146** | Three lighthouses standing on a rocky coastline at dusk. |
| strong | pbhs-counting_hard-016 | 1 | +0.125 | +0.167 | **+0.146** | Four identical telescopes set up in a row on a flat hilltop. |
| strong | pbhs-counting_hard-031 | 1 | +0.000 | +0.292 | **+0.146** | Four candy canes hanging in a row from a decorated string. |
| strong | pbhs-counting_hard-027 | 1 | +0.208 | +0.042 | **+0.125** | Five gold coins arranged in a line on black velvet. |
| full | pbh-counting_hard-017 | 2 | +0.188 | +0.042 | **+0.115** | A plate with five cookies and three brownies. |
| strong | pbhs-counting_hard-028 | 1 | -0.042 | +0.250 | **+0.104** | Four chess pieces lined up on a wooden table. |
| full | pbh-counting_hard-008 | 1 | +0.167 | +0.042 | **+0.104** | A flock of five birds flying across the sky. |
| strong | pbhs-counting_hard-018 | 1 | -0.125 | +0.333 | **+0.104** | Five oranges arranged in a small ceramic bowl. |
| strong | pbhs-counting_hard-036 | 2 | +0.062 | +0.146 | **+0.104** | Three forks and two spoons placed side by side on a folded napkin. |
| strong | pbhs-counting_hard-053 | 2 | +0.104 | +0.104 | **+0.104** | Four sandwiches and one apple in a packed lunchbox. |
| strong | pbhs-counting_hard-046 | 2 | +0.083 | +0.104 | **+0.094** | Two cameras and three binoculars laid out on a wooden table. |
| strong | pbhs-counting_hard-011 | 2 | -0.042 | +0.208 | **+0.083** | Three colorful balloons floating against a clear blue sky. |
| full | pbh-counting_hard-004 | 2 | +0.062 | +0.104 | **+0.083** | Four candles on a birthday cake. |
| full | pbh-counting_hard-068 | 2 | +0.042 | +0.125 | **+0.083** | Two snowmen standing in a snow-covered yard. |
| full | pbh-counting_hard-074 | 2 | +0.083 | +0.083 | **+0.083** | Five candles and three matches on a wooden tray. |
| strong | pbhs-counting_hard-021 | 1 | +0.042 | +0.125 | **+0.083** | Five yellow lemons placed in a row on a wooden cutting board. |
| full | pbh-counting_hard-064 | 2 | +0.083 | +0.063 | **+0.073** | Five dragonflies and two butterflies hovering near a pond. |
| full | pbh-counting_hard-035 | 2 | +0.000 | +0.146 | **+0.073** | Three soccer balls and two tennis rackets in a sports bag. |
| strong | pbhs-counting_hard-064 | 2 | +0.104 | +0.042 | **+0.073** | Three frogs and two turtles sitting on a wide lily pad. |
| full | pbh-counting_hard-020 | 3 | +0.056 | +0.069 | **+0.062** | A vase holding four sunflowers and three tulips. |
| full | pbh-counting_hard-034 | 2 | +0.021 | +0.104 | **+0.062** | Five pinecones and one acorn on a wooden table. |
| full | pbh-counting_hard-055 | 2 | +0.062 | +0.062 | **+0.062** | Five cookies and two slices of bread on a wooden cutting board. |
| full | pbh-counting_hard-056 | 1 | +0.167 | -0.042 | **+0.062** | Two scarecrows standing in a cornfield. |
| full | pbh-counting_hard-061 | 2 | +0.083 | +0.042 | **+0.062** | Three frogs and two turtles sitting on a log in a pond. |
| full | pbh-counting_hard-066 | 1 | +0.083 | +0.042 | **+0.062** | Five marbles inside a clear glass jar. |
| strong | pbhs-counting_hard-007 | 1 | +0.042 | +0.083 | **+0.062** | Four wine glasses lined up on a clean dining table. |
| full | pbh-counting_hard-007 | 3 | -0.042 | +0.167 | **+0.062** | A long table set with four plates, four wine glasses, and four forks. |
| strong | pbhs-counting_hard-004 | 1 | +0.125 | +0.000 | **+0.062** | Four orange traffic cones placed on an empty road. |
| strong | pbhs-counting_hard-068 | 2 | +0.042 | +0.062 | **+0.052** | Three rolls and two breadsticks sitting in a bread basket. |
| strong | pbhs-counting_hard-037 | 2 | +0.062 | +0.042 | **+0.052** | Four oranges and two apples arranged in a small fruit basket. |
| full | pbh-counting_hard-051 | 2 | +0.062 | +0.021 | **+0.042** | Two oranges and three pears in a wooden fruit bowl. |
| strong | pbhs-counting_hard-026 | 1 | -0.042 | +0.125 | **+0.042** | Three small picture frames standing upright on a wooden desk. |
| full | pbh-counting_hard-023 | 1 | +0.125 | -0.042 | **+0.042** | A photo of five donuts arranged in a circle. |
| full | pbh-counting_hard-031 | 2 | +0.042 | +0.042 | **+0.042** | Three eggs and two strips of bacon on a breakfast plate. |
| strong | pbhs-counting_hard-047 | 2 | +0.104 | -0.021 | **+0.042** | Five candies and two chocolates arranged in a small dish. |
| strong | pbhs-counting_hard-063 | 2 | +0.104 | -0.021 | **+0.042** | Five donuts and two cookies arranged in a small box. |
| strong | pbhs-counting_hard-051 | 2 | +0.000 | +0.083 | **+0.042** | Five marbles and two coins lying inside an open pouch. |
| strong | pbhs-counting_hard-059 | 2 | +0.021 | +0.062 | **+0.042** | Five strawberries and two blueberries on a small white plate. |
| full | pbh-counting_hard-065 | 2 | +0.104 | -0.021 | **+0.042** | Three mugs and two glasses on a kitchen table. |
| strong | pbhs-counting_hard-030 | 1 | -0.042 | +0.125 | **+0.042** | Five flowers arranged in a tall glass vase. |
| strong | pbhs-counting_hard-052 | 2 | +0.208 | -0.125 | **+0.042** | Three carrots and two potatoes laid out on a wooden cutting board. |
| strong | pbhs-counting_hard-060 | 2 | +0.021 | +0.062 | **+0.042** | Three drums and two cymbals lined up on a small stage. |
| full | pbh-counting_hard-025 | 2 | +0.042 | +0.042 | **+0.042** | Five rubber ducks floating in a row in a bathtub. |
| full | pbh-counting_hard-042 | 2 | -0.021 | +0.083 | **+0.031** | Two laptops and three notebooks on a wooden office desk. |
| strong | pbhs-counting_hard-070 | 2 | +0.083 | -0.021 | **+0.031** | Two violins and three guitars displayed in a music store. |
| strong | pbhs-counting_hard-050 | 2 | +0.062 | +0.000 | **+0.031** | Two scarves and three hats hanging on a wooden peg rack. |
| full | pbh-counting_hard-002 | 2 | -0.021 | +0.083 | **+0.031** | Five pencils lined up in a row on a desk. |
| full | pbh-counting_hard-015 | 2 | +0.062 | +0.000 | **+0.031** | Four bicycles parked next to a bike rack. |
| full | pbh-counting_hard-027 | 2 | +0.104 | -0.042 | **+0.031** | Four basketballs scattered on a wooden gym floor. |
| full | pbh-counting_hard-029 | 2 | +0.021 | +0.042 | **+0.031** | Five butterflies fluttering above a bed of flowers. |
| strong | pbhs-counting_hard-044 | 2 | +0.021 | +0.042 | **+0.031** | Three plates and two bowls stacked on an open shelf. |
| strong | pbhs-counting_hard-056 | 2 | +0.042 | +0.021 | **+0.031** | Three umbrellas and two raincoats hanging by a doorway. |
| strong | pbhs-counting_hard-057 | 2 | -0.021 | +0.083 | **+0.031** | Four kites and one balloon flying in a clear blue sky. |
| strong | pbhs-counting_hard-048 | 2 | -0.083 | +0.146 | **+0.031** | Three baseball bats and two helmets resting on a dugout bench. |
| strong | pbhs-counting_hard-032 | 1 | +0.042 | +0.000 | **+0.021** | Three masks displayed in a row on a plain white wall. |
| full | pbh-counting_hard-060 | 1 | +0.042 | +0.000 | **+0.021** | Five sand buckets lined up near the shoreline. |
| full | pbh-counting_hard-070 | 2 | +0.083 | -0.042 | **+0.021** | Three artists working at easels in a sunlit studio. |
| strong | pbhs-counting_hard-062 | 2 | +0.000 | +0.042 | **+0.021** | Two helmets and three skateboards lined up on a garage floor. |
| full | pbh-counting_hard-073 | 2 | +0.062 | -0.021 | **+0.021** | Four pumpkins and one watermelon at a farmer's market stall. |
| strong | pbhs-counting_hard-041 | 2 | +0.062 | -0.021 | **+0.021** | Four pencils and two erasers laid out in an open pencil case. |
| full | pbh-counting_hard-012 | 2 | -0.021 | +0.062 | **+0.021** | Four muffins arranged on a baking tray. |
| strong | pbhs-counting_hard-005 | 1 | -0.042 | +0.083 | **+0.021** | Three identical statues displayed on a museum floor. |
| strong | pbhs-counting_hard-019 | 1 | +0.000 | +0.042 | **+0.021** | Four toy soldiers standing upright in a line on a tabletop. |
| strong | pbhs-counting_hard-035 | 1 | +0.000 | +0.042 | **+0.021** | Three small fishbowls placed on an empty wooden desk. |
| full | pbh-counting_hard-011 | 2 | -0.042 | +0.063 | **+0.010** | A swarm of five bees hovering around a single flower. |
| full | pbh-counting_hard-033 | 2 | +0.000 | +0.021 | **+0.010** | Two helmets and three skateboards leaning against a brick wall. |
| full | pbh-counting_hard-040 | 2 | +0.021 | +0.000 | **+0.010** | Two tents pitched in a forest clearing. |
| full | pbh-counting_hard-043 | 2 | +0.000 | +0.021 | **+0.010** | Four croissants and one baguette in a bakery basket. |
| full | pbh-counting_hard-045 | 2 | +0.000 | +0.021 | **+0.010** | Five pencils and three erasers in an open pencil case. |
| strong | pbhs-counting_hard-069 | 2 | +0.062 | -0.042 | **+0.010** | Four bowls of cereal and one cup of milk on a breakfast tray. |
| strong | pbhs-counting_hard-001 | 2 | +0.042 | -0.021 | **+0.010** | Four identical mailboxes lined up along a sidewalk. |
| full | pbh-counting_hard-001 | 3 | -0.042 | +0.056 | **+0.007** | A photo of five red apples and two green pears in a wooden bowl. |
| full | pbh-counting_hard-032 | 2 | -0.125 | +0.125 | **+0.000** | Four cupcakes and two donuts on a bakery shelf. |
| strong | pbhs-counting_hard-024 | 1 | -0.042 | +0.042 | **+0.000** | Five rubber ducks floating in a clean white bathtub. |
| full | pbh-counting_hard-019 | 1 | +0.042 | -0.042 | **+0.000** | Five coins stacked on top of each other. |
| full | pbh-counting_hard-016 | 1 | +0.000 | +0.000 | **+0.000** | Five pairs of shoes lined up by the door. |
| full | pbh-counting_hard-030 | 2 | -0.021 | +0.021 | **+0.000** | Four picture frames hanging in a row above a sofa. |
| strong | pbhs-counting_hard-002 | 1 | +0.083 | -0.083 | **+0.000** | Three identical drums lined up on a small stage. |
| strong | pbhs-counting_hard-008 | 1 | -0.042 | +0.042 | **+0.000** | Three red fire hydrants spaced along an empty street. |
| strong | pbhs-counting_hard-009 | 1 | +0.083 | -0.083 | **+0.000** | Five tea candles arranged in a row on a flat wooden tray. |
| strong | pbhs-counting_hard-012 | 1 | +0.167 | -0.167 | **+0.000** | Five paint cans lined up on a workshop floor. |
| strong | pbhs-counting_hard-013 | 1 | +0.000 | +0.000 | **+0.000** | Four pillows arranged neatly in a row on a wide bed. |
| strong | pbhs-counting_hard-014 | 1 | +0.000 | +0.000 | **+0.000** | Three identical trophies displayed on a polished mantel. |
| strong | pbhs-counting_hard-049 | 2 | +0.083 | -0.083 | **+0.000** | Four lemons and one lime in a wicker basket. |
| full | pbh-counting_hard-003 | 2 | -0.021 | +0.021 | **-0.000** | Four ducks swimming in a pond, with two ducklings nearby. |
| full | pbh-counting_hard-049 | 1 | +0.125 | -0.125 | **-0.000** | Five hats hanging on hooks by a front door. |
| full | pbh-counting_hard-053 | 2 | -0.021 | +0.000 | **-0.010** | Three pillows and one teddy bear on a neatly made bed. |
| full | pbh-counting_hard-072 | 2 | -0.021 | +0.000 | **-0.010** | Two trumpets and three saxophones displayed in a music store. |
| strong | pbhs-counting_hard-042 | 2 | +0.083 | -0.104 | **-0.010** | Two boots and three sneakers lined up on a shoe rack. |
| strong | pbhs-counting_hard-067 | 2 | -0.021 | +0.000 | **-0.010** | Five gummy bears and two jellybeans inside a clear candy jar. |
| full | pbh-counting_hard-005 | 2 | -0.083 | +0.062 | **-0.010** | A pile of exactly five oranges next to three lemons. |
| full | pbh-counting_hard-052 | 2 | +0.062 | -0.083 | **-0.010** | Four onions and two garlic bulbs on a kitchen counter. |
| full | pbh-counting_hard-054 | 2 | -0.021 | +0.000 | **-0.010** | Two umbrellas and three raincoats hanging by a doorway. |
| strong | pbhs-counting_hard-039 | 2 | +0.062 | -0.083 | **-0.010** | Five candles and two cupcakes on a serving platter. |
| strong | pbhs-counting_hard-045 | 2 | +0.021 | -0.042 | **-0.010** | Four croissants and one muffin in a bakery basket. |
| full | pbh-counting_hard-010 | 2 | +0.042 | -0.062 | **-0.010** | Four bottles of water lined up on a shelf. |
| full | pbh-counting_hard-013 | 2 | +0.000 | -0.021 | **-0.010** | Three cats and two dogs sitting together on a couch. |
| full | pbh-counting_hard-037 | 2 | -0.021 | +0.000 | **-0.010** | Four chairs arranged around a small dining table. |
| strong | pbhs-counting_hard-040 | 2 | -0.042 | +0.021 | **-0.010** | Three pillows and two stuffed bears on a clean bed. |
| strong | pbhs-counting_hard-066 | 2 | -0.062 | +0.042 | **-0.010** | Two skis and three snowboards leaning against a wooden wall. |
| full | pbh-counting_hard-050 | 2 | +0.021 | -0.042 | **-0.010** | Four seashells lined up on a sandy beach. |
| full | pbh-counting_hard-048 | 2 | +0.000 | -0.042 | **-0.021** | Two telescopes set up on a hilltop at night. |
| full | pbh-counting_hard-071 | 2 | +0.042 | -0.083 | **-0.021** | Three apples and two bananas in a child's lunchbox. |
| strong | pbhs-counting_hard-025 | 1 | +0.000 | -0.042 | **-0.021** | Four colored ribbons tied to a single bare tree branch. |
| strong | pbhs-counting_hard-033 | 1 | -0.083 | +0.042 | **-0.021** | Five postcards stacked neatly on a coffee table. |
| strong | pbhs-counting_hard-055 | 2 | -0.021 | -0.021 | **-0.021** | Five letters and two stamps on a wooden desk. |
| full | pbh-counting_hard-022 | 2 | +0.083 | -0.125 | **-0.021** | Five surfboards stuck upright in the sand on a beach. |
| full | pbh-counting_hard-024 | 2 | -0.042 | -0.021 | **-0.031** | Four kites flying in a clear blue sky with no clouds. |
| full | pbh-counting_hard-058 | 2 | -0.042 | -0.021 | **-0.031** | Three rocking chairs lined up on a wooden front porch. |
| strong | pbhs-counting_hard-020 | 1 | -0.042 | -0.042 | **-0.042** | Three watering cans lined up on a small garden bench. |
| full | pbh-counting_hard-014 | 2 | -0.146 | +0.062 | **-0.042** | A chess board with five pawns and two knights placed on it. |
| full | pbh-counting_hard-041 | 2 | -0.062 | -0.021 | **-0.042** | Three hamburgers and two hot dogs on a restaurant counter. |
| full | pbh-counting_hard-018 | 2 | +0.000 | -0.083 | **-0.042** | Five children playing in a sandbox. |
| full | pbh-counting_hard-067 | 1 | -0.042 | -0.042 | **-0.042** | Three antique clocks displayed on a museum shelf. |
| strong | pbhs-counting_hard-006 | 2 | -0.104 | +0.000 | **-0.052** | Five orange pumpkins arranged in a row on a wooden porch. |
| strong | pbhs-counting_hard-058 | 2 | +0.000 | -0.104 | **-0.052** | Two saxophones and three trumpets displayed in a music store. |
| full | pbh-counting_hard-044 | 2 | -0.042 | -0.062 | **-0.052** | Three lemons and two limes in a small ceramic bowl. |
| strong | pbhs-counting_hard-010 | 1 | -0.042 | -0.083 | **-0.062** | Four bookends lined up on a long empty shelf. |
| strong | pbhs-counting_hard-043 | 2 | -0.083 | -0.042 | **-0.062** | Five eggs and two slices of bread laid out on a kitchen counter. |
| strong | pbhs-counting_hard-054 | 2 | +0.000 | -0.125 | **-0.062** | Two surfboards and three towels arranged on a sandy beach. |
| strong | pbhs-counting_hard-065 | 2 | +0.042 | -0.167 | **-0.062** | Four ducks and one swan in a small pond. |
| full | pbh-counting_hard-075 | 2 | -0.125 | +0.000 | **-0.063** | Two helicopters and three planes parked inside a hangar. |
| full | pbh-counting_hard-036 | 1 | +0.083 | -0.250 | **-0.083** | Two motorcycles parked on a city sidewalk. |
| full | pbh-counting_hard-039 | 2 | -0.062 | -0.104 | **-0.083** | Five buttons running down the front of a denim shirt. |
| full | pbh-counting_hard-059 | 1 | -0.167 | +0.000 | **-0.083** | Two large sandcastles on a sandy beach. |
| strong | pbhs-counting_hard-015 | 1 | -0.083 | -0.083 | **-0.083** | Five lightbulbs hanging in a row from a single cord. |
| strong | pbhs-counting_hard-022 | 1 | -0.042 | -0.125 | **-0.083** | Four mugs hanging from hooks under a kitchen cabinet. |
| full | pbh-counting_hard-006 | 2 | -0.062 | -0.125 | **-0.094** | Five sheep grazing in a field; no shepherd is visible. |
| full | pbh-counting_hard-063 | 2 | -0.042 | -0.146 | **-0.094** | Two crows and three sparrows perched on a wooden fence. |
| strong | pbhs-counting_hard-038 | 2 | -0.146 | -0.062 | **-0.104** | Two coffee mugs and three teacups arranged on a wooden serving tray. |
| full | pbh-counting_hard-026 | 2 | -0.104 | -0.125 | **-0.115** | Three guitars hanging on the wall in a music store. |
| full | pbh-counting_hard-028 | 2 | +0.000 | -0.229 | **-0.115** | Two slices of pizza on a paper plate. |
| strong | pbhs-counting_hard-003 | 2 | -0.062 | -0.167 | **-0.115** | Five paper lanterns hanging in a row from a wooden beam. |
| strong | pbhs-counting_hard-034 | 1 | -0.208 | -0.042 | **-0.125** | Four ice cream cones standing upright on a striped tray. |
| strong | pbhs-counting_hard-023 | 1 | +0.042 | -0.292 | **-0.125** | Three identical white vases placed on a long shelf. |
| full | pbh-counting_hard-009 | 2 | -0.208 | -0.063 | **-0.135** | Five mushrooms growing on a forest floor next to a fallen log. |
| full | pbh-counting_hard-038 | 1 | -0.125 | -0.167 | **-0.146** | Three vases lined up on a windowsill. |
| full | pbh-counting_hard-046 | 2 | -0.146 | -0.146 | **-0.146** | Four boats moored at a wooden dock. |
| strong | pbhs-counting_hard-017 | 1 | -0.292 | +0.000 | **-0.146** | Three soccer balls placed on a freshly mown lawn. |
| full | pbh-counting_hard-069 | 1 | -0.417 | +0.125 | **-0.146** | Four life jackets hanging on a wall in a boathouse. |
| strong | pbhs-counting_hard-061 | 2 | -0.188 | -0.104 | **-0.146** | Four bottles and one jar lined up on a kitchen counter. |
| full | pbh-counting_hard-062 | 2 | -0.229 | -0.188 | **-0.208** | Four squirrels and one chipmunk in the branches of a tree. |
| strong | pbhs-counting_hard-029 | 1 | -0.250 | -0.333 | **-0.292** | Three red apples placed in a row on a clean white plate. |

## count_x_color (130 prompts)

Distribution of avg_Δ:

  - -1.00 ≤ Δ < -0.10: 4
  - -0.10 ≤ Δ < -0.05: 22
  - -0.05 ≤ Δ < +0.00: 31
  - +0.00 ≤ Δ < +0.05: 50
  - +0.05 ≤ Δ < +0.10: 10
  - +0.10 ≤ Δ < +1.00: 13

| ds | id | n_q | Δ_L | Δ_XL | avg_Δ | prompt |
|---|---|---|---|---|---|---|
| full | pbh-count_x_color-029 | 2 | +0.042 | +0.312 | **+0.177** | Four blue marbles and one green marble inside a clear glass jar. |
| full | pbh-count_x_color-052 | 2 | +0.104 | +0.167 | **+0.135** | Four red roses and one white rose in a long-stemmed bouquet. |
| full | pbh-count_x_color-055 | 2 | +0.146 | +0.125 | **+0.135** | Four yellow lemons and one green lime in a wicker basket. |
| full | pbh-count_x_color-025 | 2 | +0.167 | +0.083 | **+0.125** | Three red M&Ms and two yellow M&Ms scattered on a table. |
| strong | pbhs-count_x_color-055 | 2 | +0.042 | +0.208 | **+0.125** | Two pink balloons and three white balloons floating at a baby shower. |
| strong | pbhs-count_x_color-005 | 2 | +0.083 | +0.167 | **+0.125** | Three black cats and two orange cats sitting on a porch. |
| strong | pbhs-count_x_color-030 | 2 | +0.104 | +0.146 | **+0.125** | Four orange pumpkins and one white pumpkin arranged on a porch. |
| full | pbh-count_x_color-045 | 2 | +0.083 | +0.146 | **+0.115** | Three white pillows and two navy blue pillows arranged on a couch. |
| full | pbh-count_x_color-015 | 2 | +0.062 | +0.167 | **+0.115** | Three black cats and one orange cat sitting in front of a fireplace. |
| strong | pbhs-count_x_color-050 | 2 | +0.062 | +0.167 | **+0.115** | Four green toy soldiers and one red toy soldier on a wooden floor. |
| full | pbh-count_x_color-018 | 2 | +0.063 | +0.146 | **+0.104** | Four pink macarons and three green macarons stacked on a small plate. |
| full | pbh-count_x_color-038 | 2 | +0.083 | +0.125 | **+0.104** | Four green leaves and one red leaf scattered on a stone path. |
| full | pbh-count_x_color-039 | 2 | +0.062 | +0.146 | **+0.104** | Two black guitars and three brown guitars hanging on a music store wall. |
| strong | pbhs-count_x_color-002 | 2 | +0.042 | +0.146 | **+0.094** | Three red balloons and two white balloons floating against a clear sky. |
| strong | pbhs-count_x_color-031 | 2 | +0.042 | +0.125 | **+0.083** | Two pink flamingos and three white flamingos standing in a shallow pool. |
| strong | pbhs-count_x_color-042 | 2 | +0.042 | +0.125 | **+0.083** | Four green books and one red book lined up on a wooden shelf. |
| full | pbh-count_x_color-004 | 2 | +0.042 | +0.104 | **+0.073** | Five purple flowers and two orange flowers in the same vase. |
| full | pbh-count_x_color-030 | 2 | -0.042 | +0.188 | **+0.073** | Five white horses and one brown horse in a grassy pasture. |
| strong | pbhs-count_x_color-033 | 2 | +0.062 | +0.083 | **+0.073** | Three green bottles and two yellow bottles standing on a shelf. |
| full | pbh-count_x_color-023 | 2 | +0.000 | +0.125 | **+0.063** | Two blue mugs and three yellow mugs lined up on a kitchen shelf. |
| strong | pbhs-count_x_color-038 | 2 | +0.167 | -0.042 | **+0.062** | Four blue blocks and one yellow block placed on a wooden floor. |
| strong | pbhs-count_x_color-019 | 2 | +0.062 | +0.062 | **+0.062** | Two black cars and three white cars parked in a parking lot. |
| strong | pbhs-count_x_color-009 | 2 | +0.021 | +0.083 | **+0.052** | Three white pillows and two red pillows arranged on a couch. |
| strong | pbhs-count_x_color-011 | 2 | +0.042 | +0.042 | **+0.042** | Two red apples and three green apples in a wooden fruit bowl. |
| strong | pbhs-count_x_color-025 | 2 | -0.021 | +0.104 | **+0.042** | Three white sneakers and two red sneakers lined up on a shoe rack. |
| strong | pbhs-count_x_color-053 | 2 | +0.042 | +0.042 | **+0.042** | Three blue plates and two yellow plates stacked on a kitchen counter. |
| full | pbh-count_x_color-046 | 2 | +0.083 | +0.000 | **+0.042** | Four white candles and one black candle on top of a birthday cake. |
| strong | pbhs-count_x_color-010 | 2 | +0.104 | -0.021 | **+0.042** | Five green leaves and one orange leaf scattered on a wooden table. |
| strong | pbhs-count_x_color-049 | 2 | +0.083 | +0.000 | **+0.042** | Three red roses and two yellow roses arranged in a glass vase. |
| full | pbh-count_x_color-017 | 2 | +0.042 | +0.042 | **+0.042** | Two green soldiers and four blue soldiers as toy figurines on a wooden floor. |
| strong | pbhs-count_x_color-020 | 2 | +0.021 | +0.062 | **+0.042** | Five red bricks and two yellow bricks stacked on a workbench. |
| full | pbh-count_x_color-060 | 2 | +0.083 | -0.021 | **+0.031** | Three yellow pencils and two black pens lying together on a desk. |
| full | pbh-count_x_color-002 | 2 | -0.021 | +0.083 | **+0.031** | Four red balloons and one green balloon floating against a clear sky. |
| full | pbh-count_x_color-016 | 2 | +0.000 | +0.062 | **+0.031** | Five red roses and two white roses in a bouquet. |
| full | pbh-count_x_color-049 | 2 | +0.000 | +0.062 | **+0.031** | Four yellow pencils and one blue pencil in a glass cup. |
| strong | pbhs-count_x_color-040 | 2 | +0.042 | +0.021 | **+0.031** | Five red beads and two white beads strung on a single thread. |
| strong | pbhs-count_x_color-054 | 2 | -0.083 | +0.146 | **+0.031** | Four black umbrellas and one red umbrella standing by a doorway. |
| full | pbh-count_x_color-007 | 2 | +0.021 | +0.042 | **+0.031** | Five donuts on a tray: three with pink frosting and two with chocolate frosting. |
| full | pbh-count_x_color-035 | 2 | +0.062 | +0.000 | **+0.031** | Two red tulips and three yellow tulips growing in a garden bed. |
| full | pbh-count_x_color-050 | 2 | +0.104 | -0.042 | **+0.031** | Five white tulips and two pink tulips in a small bouquet. |
| strong | pbhs-count_x_color-007 | 2 | +0.083 | -0.021 | **+0.031** | Two purple flowers and three yellow flowers in a garden bed. |
| full | pbh-count_x_color-010 | 2 | -0.021 | +0.062 | **+0.021** | Three orange koi fish and two white koi fish swimming in a pond. |
| strong | pbhs-count_x_color-017 | 2 | +0.125 | -0.083 | **+0.021** | Three red gummy bears and two green gummy bears arranged on a plate. |
| full | pbh-count_x_color-036 | 2 | -0.083 | +0.125 | **+0.021** | Five brown shoes and one white shoe on a shoe rack by the door. |
| full | pbh-count_x_color-006 | 2 | +0.146 | -0.104 | **+0.021** | Two green books stacked on top of three red books. |
| strong | pbhs-count_x_color-015 | 2 | +0.021 | +0.021 | **+0.021** | Two red candles and three blue candles standing on a wooden tray. |
| strong | pbhs-count_x_color-061 | 2 | +0.021 | +0.021 | **+0.021** | Three orange tulips and two purple tulips in a small flower bed. |
| strong | pbhs-count_x_color-044 | 2 | -0.021 | +0.062 | **+0.021** | Five red dots and two blue dots painted on a white canvas. |
| strong | pbhs-count_x_color-046 | 2 | -0.042 | +0.083 | **+0.021** | Four white candles and one red candle on top of a birthday cake. |
| full | pbh-count_x_color-022 | 2 | +0.062 | -0.021 | **+0.021** | Four black umbrellas and one yellow umbrella in an umbrella stand. |
| full | pbh-count_x_color-057 | 2 | +0.104 | -0.062 | **+0.021** | Three red kites and two yellow kites flying high in the sky. |
| full | pbh-count_x_color-056 | 2 | +0.021 | +0.021 | **+0.021** | Two black motorcycles and three silver motorcycles parked side by side. |
| full | pbh-count_x_color-008 | 2 | +0.083 | -0.062 | **+0.010** | Three white candles and two black candles in a row on a stone shelf. |
| strong | pbhs-count_x_color-043 | 2 | +0.021 | +0.000 | **+0.010** | Two yellow lemons and three green limes in a fruit bowl. |
| strong | pbhs-count_x_color-012 | 2 | +0.062 | -0.042 | **+0.010** | Four blue mugs and one orange mug hanging on hooks under a cabinet. |
| full | pbh-count_x_color-014 | 2 | +0.062 | -0.042 | **+0.010** | Two purple grapes and four green grapes scattered on a wooden cutting board. |
| strong | pbhs-count_x_color-024 | 2 | +0.021 | +0.000 | **+0.010** | Five red ribbons and two blue ribbons tied around a single gift box. |
| strong | pbhs-count_x_color-066 | 2 | +0.000 | +0.021 | **+0.010** | Four blue gummy candies and one red gummy candy in a candy bag. |
| strong | pbhs-count_x_color-041 | 2 | -0.083 | +0.104 | **+0.010** | Three orange flowers and two purple flowers in a backyard garden. |
| strong | pbhs-count_x_color-056 | 2 | -0.083 | +0.104 | **+0.010** | Five red apples and two yellow apples placed in a wooden bowl. |
| strong | pbhs-count_x_color-065 | 2 | +0.042 | -0.021 | **+0.010** | Three white cupcakes and two pink cupcakes on a serving plate. |
| full | pbh-count_x_color-001 | 3 | +0.042 | -0.028 | **+0.007** | Three blue cars and two yellow cars parked in a row. |
| full | pbh-count_x_color-011 | 2 | +0.021 | -0.021 | **+0.000** | Four green frogs and one yellow frog sitting on a lily pad. |
| strong | pbhs-count_x_color-057 | 2 | -0.021 | +0.021 | **+0.000** | Three green frogs and two yellow frogs sitting on a wide lily pad. |
| full | pbh-count_x_color-031 | 2 | +0.062 | -0.062 | **+0.000** | Three black socks and two white socks on a clothesline. |
| full | pbh-count_x_color-047 | 2 | +0.000 | +0.000 | **+0.000** | Two red sneakers and three white sneakers on a shoe rack. |
| full | pbh-count_x_color-051 | 2 | +0.000 | +0.000 | **+0.000** | Three black cows and two brown cows grazing in a pasture. |
| strong | pbhs-count_x_color-006 | 2 | -0.021 | +0.021 | **+0.000** | Five red roses and two green leaves arranged in a vase. |
| strong | pbhs-count_x_color-026 | 2 | +0.042 | -0.042 | **+0.000** | Four green peppers and one red pepper on a wooden cutting board. |
| strong | pbhs-count_x_color-027 | 2 | +0.083 | -0.083 | **+0.000** | Two black cupcakes and three white cupcakes on a serving tray. |
| strong | pbhs-count_x_color-028 | 2 | +0.000 | +0.000 | **+0.000** | Five purple grapes and two green grapes arranged on a plate. |
| strong | pbhs-count_x_color-048 | 2 | +0.000 | +0.000 | **+0.000** | Five orange goldfish and two black guppies swimming in an aquarium. |
| strong | pbhs-count_x_color-064 | 2 | +0.000 | +0.000 | **+0.000** | Five purple grapes and two red grapes arranged on a small plate. |
| full | pbh-count_x_color-033 | 2 | +0.062 | -0.062 | **-0.000** | Two yellow bananas and three green bananas in a bunch. |
| strong | pbhs-count_x_color-016 | 2 | +0.000 | -0.021 | **-0.010** | Four yellow umbrellas and one black umbrella in a tall umbrella stand. |
| strong | pbhs-count_x_color-034 | 2 | +0.042 | -0.062 | **-0.010** | Four white cats and one black cat sitting together on a striped rug. |
| strong | pbhs-count_x_color-036 | 2 | +0.042 | -0.062 | **-0.010** | Five purple eggs and two yellow eggs lying in a wicker basket. |
| strong | pbhs-count_x_color-037 | 2 | -0.062 | +0.042 | **-0.010** | Three white hats and two red hats hanging on a coat rack. |
| strong | pbhs-count_x_color-001 | 2 | -0.125 | +0.104 | **-0.010** | Four red marbles and one green marble in a clear glass jar. |
| strong | pbhs-count_x_color-014 | 2 | +0.000 | -0.021 | **-0.010** | Five black beans and two white beans inside a small ceramic bowl. |
| full | pbh-count_x_color-058 | 2 | +0.042 | -0.062 | **-0.010** | Four white doves and one grey pigeon perched on a rooftop. |
| strong | pbhs-count_x_color-045 | 2 | +0.021 | -0.042 | **-0.010** | Three pink cookies and two green cookies arranged on a small plate. |
| full | pbh-count_x_color-012 | 2 | -0.042 | +0.000 | **-0.021** | Two red bell peppers and three yellow bell peppers on a cutting board. |
| full | pbh-count_x_color-041 | 2 | +0.104 | -0.146 | **-0.021** | Two white sails and three red sails on a row of boats at sea. |
| full | pbh-count_x_color-053 | 2 | +0.083 | -0.125 | **-0.021** | Two white teacups and three blue teacups arranged on a serving tray. |
| full | pbh-count_x_color-003 | 2 | +0.021 | -0.062 | **-0.021** | Two black dogs and three white dogs in a grassy park. |
| strong | pbhs-count_x_color-029 | 2 | +0.062 | -0.104 | **-0.021** | Three red socks and two blue socks pinned on a clothesline. |
| full | pbh-count_x_color-020 | 2 | +0.021 | -0.062 | **-0.021** | Four white sailboats and two red sailboats on a calm sea at sunset. |
| full | pbh-count_x_color-054 | 2 | -0.125 | +0.083 | **-0.021** | Three brown squirrels and one grey squirrel running across a lawn. |
| strong | pbhs-count_x_color-013 | 2 | +0.083 | -0.125 | **-0.021** | Three white tulips and two purple tulips in a tall vase. |
| strong | pbhs-count_x_color-047 | 2 | -0.167 | +0.125 | **-0.021** | Two black ravens and three white doves perched on a wooden fence. |
| strong | pbhs-count_x_color-067 | 2 | -0.042 | +0.000 | **-0.021** | Two black mittens and three red mittens laid out on a wooden table. |
| strong | pbhs-count_x_color-032 | 2 | -0.021 | -0.042 | **-0.031** | Five red lollipops and two yellow lollipops in a candy jar. |
| full | pbh-count_x_color-019 | 2 | +0.063 | -0.125 | **-0.031** | Three red apples and two yellow apples in a single basket. |
| full | pbh-count_x_color-024 | 2 | -0.042 | -0.021 | **-0.031** | Five green grapes and one purple grape on a small white plate. |
| strong | pbhs-count_x_color-018 | 2 | +0.125 | -0.188 | **-0.031** | Four green pears and one red apple in a wicker basket. |
| strong | pbhs-count_x_color-070 | 2 | +0.062 | -0.125 | **-0.031** | Four pink ribbons and one blue ribbon tied around a single gift box. |
| strong | pbhs-count_x_color-008 | 2 | +0.021 | -0.083 | **-0.031** | Four pink balloons and one blue balloon at a child's birthday party. |
| full | pbh-count_x_color-026 | 2 | -0.063 | +0.000 | **-0.031** | Four white sheep and one black sheep grazing in a green field. |
| strong | pbhs-count_x_color-062 | 2 | +0.083 | -0.167 | **-0.042** | Four red bell peppers and one yellow bell pepper in a wicker basket. |
| strong | pbhs-count_x_color-063 | 2 | +0.125 | -0.208 | **-0.042** | Two blue boats and three white boats on a calm blue lake. |
| strong | pbhs-count_x_color-058 | 2 | -0.062 | -0.021 | **-0.042** | Four white sails and one red sail on small boats at a marina. |
| full | pbh-count_x_color-005 | 2 | -0.083 | +0.000 | **-0.042** | Three yellow rubber ducks and one blue rubber duck in a bathtub. |
| full | pbh-count_x_color-048 | 2 | -0.167 | +0.083 | **-0.042** | Three yellow taxi cabs and two black sedans driving through an intersection. |
| strong | pbhs-count_x_color-023 | 2 | -0.021 | -0.083 | **-0.052** | Two yellow tennis balls and three green tennis balls on a tennis court. |
| full | pbh-count_x_color-059 | 2 | -0.062 | -0.042 | **-0.052** | Two red pawns and three white pawns set up on a chess board. |
| strong | pbhs-count_x_color-003 | 2 | -0.146 | +0.042 | **-0.052** | Two yellow ducks and three white ducks floating on a small pond. |
| strong | pbhs-count_x_color-035 | 2 | -0.062 | -0.042 | **-0.052** | Two red kites and three blue kites flying high in a clear sky. |
| strong | pbhs-count_x_color-060 | 2 | -0.042 | -0.063 | **-0.052** | Five red M&Ms and two blue M&Ms scattered on a wooden table. |
| full | pbh-count_x_color-034 | 2 | +0.021 | -0.146 | **-0.062** | Four white lilies and one pink lily arranged in a tall vase. |
| full | pbh-count_x_color-021 | 2 | -0.021 | -0.104 | **-0.062** | Three red flowers and two white flowers in the same vase. |
| strong | pbhs-count_x_color-004 | 2 | -0.083 | -0.042 | **-0.062** | Four blue M&Ms and one yellow M&M arranged on a small dish. |
| strong | pbhs-count_x_color-039 | 2 | -0.062 | -0.062 | **-0.062** | Two black pencils and three red pencils standing in a glass cup. |
| full | pbh-count_x_color-032 | 2 | -0.125 | +0.000 | **-0.063** | Four red ornaments and two gold ornaments hanging on a Christmas tree. |
| strong | pbhs-count_x_color-051 | 2 | -0.146 | +0.000 | **-0.073** | Two black notebooks and three red notebooks stacked on a desk. |
| full | pbh-count_x_color-028 | 2 | -0.083 | -0.063 | **-0.073** | Three red dice and two white dice on a felt board. |
| full | pbh-count_x_color-009 | 2 | -0.146 | -0.021 | **-0.083** | Two red mugs and four blue mugs hanging on hooks under a kitchen cabinet. |
| full | pbh-count_x_color-027 | 2 | -0.104 | -0.062 | **-0.083** | Two orange fish and three yellow fish swimming in a clear aquarium. |
| full | pbh-count_x_color-042 | 2 | -0.021 | -0.146 | **-0.083** | Three black knight chess pieces and two white knight chess pieces on a board. |
| strong | pbhs-count_x_color-022 | 2 | -0.021 | -0.146 | **-0.083** | Four pink macarons and one green macaron on a small plate. |
| full | pbh-count_x_color-013 | 2 | -0.042 | -0.125 | **-0.083** | Three pink flamingos and two white swans in a shallow pond. |
| strong | pbhs-count_x_color-059 | 2 | -0.042 | -0.125 | **-0.083** | Two red M&Ms and three green M&Ms scattered on a small saucer. |
| strong | pbhs-count_x_color-068 | 2 | +0.083 | -0.250 | **-0.083** | Five yellow buttons and two black buttons sewn on a piece of cloth. |
| full | pbh-count_x_color-043 | 2 | -0.208 | +0.021 | **-0.094** | Four brown horses and one grey horse standing in a stable. |
| strong | pbhs-count_x_color-021 | 2 | -0.062 | -0.125 | **-0.094** | Three orange goldfish and two black goldfish swimming in a small bowl. |
| full | pbh-count_x_color-037 | 2 | -0.125 | -0.062 | **-0.094** | Three red toy robots and two blue toy robots displayed on a shelf. |
| strong | pbhs-count_x_color-069 | 2 | -0.104 | -0.104 | **-0.104** | Three orange goldfish and two white goldfish in a clear fishbowl. |
| full | pbh-count_x_color-040 | 2 | -0.188 | -0.042 | **-0.115** | Four yellow chicks and one brown chick walking through a small pen. |
| full | pbh-count_x_color-044 | 2 | -0.104 | -0.188 | **-0.146** | Two silver coins and three gold coins resting on a piece of velvet. |
| strong | pbhs-count_x_color-052 | 2 | -0.229 | -0.062 | **-0.146** | Five red ladybugs and two black ladybugs resting on a green leaf. |

## composition_layout (105 prompts)

Distribution of avg_Δ:

  - -1.00 ≤ Δ < -0.10: 16
  - -0.10 ≤ Δ < -0.05: 14
  - -0.05 ≤ Δ < +0.00: 22
  - +0.00 ≤ Δ < +0.05: 28
  - +0.05 ≤ Δ < +0.10: 16
  - +0.10 ≤ Δ < +1.00: 9

| ds | id | n_q | Δ_L | Δ_XL | avg_Δ | prompt |
|---|---|---|---|---|---|---|
| strong | pbhs-composition_layout-018 | 2 | +0.333 | +0.167 | **+0.250** | A dog sitting between two children on a grassy lawn. |
| strong | pbhs-composition_layout-010 | 1 | -0.083 | +0.458 | **+0.187** | A child walking to the right of an adult down a quiet street. |
| strong | pbhs-composition_layout-067 | 1 | +0.083 | +0.292 | **+0.187** | A girl sitting to the left of a boy on a school bus seat. |
| strong | pbhs-composition_layout-055 | 1 | +0.208 | +0.125 | **+0.167** | A coffee mug between a notebook and a phone on an office desk. |
| strong | pbhs-composition_layout-054 | 2 | +0.146 | +0.125 | **+0.135** | A picnic basket between two folding chairs on a grassy lawn. |
| full | pbh-composition_layout-028 | 2 | +0.146 | +0.125 | **+0.135** | An umbrella to the left of a beach chair with a towel folded on the right. |
| strong | pbhs-composition_layout-009 | 2 | +0.104 | +0.125 | **+0.115** | A blue car parked to the left of a red car at a curb. |
| full | pbh-composition_layout-002 | 3 | +0.111 | +0.111 | **+0.111** | A red bowl to the left of a green bowl, with a fork between them. |
| full | pbh-composition_layout-013 | 2 | +0.083 | +0.125 | **+0.104** | Two books on the left and three books on the right of a vase on a shelf. |
| strong | pbhs-composition_layout-005 | 2 | -0.021 | +0.208 | **+0.094** | A teapot placed between two teacups on a wooden tray. |
| full | pbh-composition_layout-024 | 2 | -0.062 | +0.250 | **+0.094** | A bird perched on the top branch of a tree with a cat sitting under it. |
| strong | pbhs-composition_layout-068 | 1 | +0.083 | +0.083 | **+0.083** | An empty cup to the left of a full cup on a coffee table. |
| strong | pbhs-composition_layout-069 | 2 | +0.229 | -0.062 | **+0.083** | A red cup placed between two blue cups on a wooden tray. |
| full | pbh-composition_layout-017 | 2 | +0.083 | +0.062 | **+0.073** | Three paintings hung in a row on a wall, with the largest in the middle. |
| full | pbh-composition_layout-025 | 2 | +0.104 | +0.042 | **+0.073** | A sandwich on a plate with a glass of milk to its right. |
| full | pbh-composition_layout-008 | 2 | +0.271 | -0.146 | **+0.063** | A vase of flowers placed exactly between two framed pictures on a mantel. |
| full | pbh-composition_layout-005 | 2 | +0.062 | +0.062 | **+0.062** | A laptop on a desk, with a notebook to its left and a mug to its right. |
| full | pbh-composition_layout-012 | 2 | -0.062 | +0.188 | **+0.062** | A teapot in the center of a tray with three teacups arranged around it. |
| strong | pbhs-composition_layout-041 | 2 | -0.021 | +0.146 | **+0.062** | A rabbit sitting between two carrots on a green lawn. |
| strong | pbhs-composition_layout-061 | 2 | +0.083 | +0.042 | **+0.062** | A red boat between two white boats moored in a small harbor. |
| full | pbh-composition_layout-010 | 3 | +0.069 | +0.056 | **+0.062** | A child holding a balloon while standing under a tree. |
| strong | pbhs-composition_layout-046 | 1 | +0.042 | +0.083 | **+0.062** | A bicycle parked to the left of a tree in a sunny park. |
| strong | pbhs-composition_layout-034 | 2 | -0.021 | +0.125 | **+0.052** | A fork on the left, a knife in the middle, and a spoon on the right of a folded napkin. |
| strong | pbhs-composition_layout-053 | 2 | +0.062 | +0.042 | **+0.052** | A small flower pot between two larger flower pots on a windowsill. |
| strong | pbhs-composition_layout-048 | 2 | -0.021 | +0.125 | **+0.052** | A child sitting between two parents on a wooden park bench. |
| full | pbh-composition_layout-034 | 2 | +0.021 | +0.062 | **+0.042** | A cyclist on the left side of a crosswalk and a pedestrian on the right side. |
| strong | pbhs-composition_layout-033 | 2 | +0.042 | +0.042 | **+0.042** | A teacher standing between two students at a chalkboard. |
| strong | pbhs-composition_layout-063 | 1 | +0.000 | +0.083 | **+0.042** | A toy car to the left of a toy truck on a child's bedroom floor. |
| full | pbh-composition_layout-022 | 2 | -0.062 | +0.146 | **+0.042** | A laptop in front of a monitor with a keyboard between them on a desk. |
| full | pbh-composition_layout-006 | 2 | +0.062 | +0.021 | **+0.042** | Two pillows on a bed with a folded blanket between them. |
| strong | pbhs-composition_layout-011 | 2 | +0.062 | +0.021 | **+0.042** | A picture frame placed between two lit candles on a mantel. |
| full | pbh-composition_layout-032 | 2 | +0.125 | -0.062 | **+0.031** | A bookshelf with three books on the top shelf and two books on the bottom shelf. |
| strong | pbhs-composition_layout-024 | 2 | +0.000 | +0.062 | **+0.031** | A guitar standing between two amplifiers on a small stage. |
| strong | pbhs-composition_layout-002 | 2 | -0.062 | +0.125 | **+0.031** | A small dog sitting to the right of a large cat on a wooden floor. |
| full | pbh-composition_layout-019 | 2 | +0.021 | +0.042 | **+0.031** | A clock hanging above a fireplace with two photos placed below. |
| strong | pbhs-composition_layout-039 | 2 | +0.104 | -0.042 | **+0.031** | A bottle to the left of two glasses on a bar counter. |
| strong | pbhs-composition_layout-065 | 2 | +0.000 | +0.062 | **+0.031** | A hot air balloon floating between two clouds in a clear sky. |
| strong | pbhs-composition_layout-007 | 3 | -0.042 | +0.097 | **+0.028** | A lamp on the left, a clock in the middle, and a vase on the right of a desk. |
| full | pbh-composition_layout-007 | 2 | +0.083 | -0.042 | **+0.021** | Three apples arranged in a triangle on a wooden table. |
| full | pbh-composition_layout-026 | 2 | +0.021 | +0.021 | **+0.021** | An ice cream cone with two scoops of ice cream stacked on top. |
| full | pbh-composition_layout-023 | 2 | -0.021 | +0.062 | **+0.021** | A kettle on a stove with two cups beside it on the counter. |
| full | pbh-composition_layout-027 | 2 | +0.062 | -0.021 | **+0.021** | Two ducks in front of a pond and one duck behind it. |
| strong | pbhs-composition_layout-043 | 1 | -0.083 | +0.125 | **+0.021** | An empty plate between two filled plates on a long dining table. |
| full | pbh-composition_layout-004 | 2 | +0.021 | +0.021 | **+0.021** | A small dog standing in front of a large horse in a barn. |
| strong | pbhs-composition_layout-013 | 1 | +0.000 | +0.042 | **+0.021** | A bottle of water to the left of a sandwich on a picnic blanket. |
| strong | pbhs-composition_layout-014 | 2 | +0.042 | +0.000 | **+0.021** | A small cat sitting between two pillows on a couch. |
| strong | pbhs-composition_layout-052 | 1 | +0.042 | +0.000 | **+0.021** | A baseball bat to the left of a baseball glove on a wooden bench. |
| full | pbh-composition_layout-018 | 2 | +0.042 | +0.000 | **+0.021** | Two lamps flanking a long sofa in a modern living room. |
| strong | pbhs-composition_layout-020 | 2 | +0.146 | -0.125 | **+0.010** | A lighthouse on the left and a sailboat on the right of a beach. |
| strong | pbhs-composition_layout-008 | 2 | +0.062 | -0.042 | **+0.010** | Three cookies between two glasses of milk on a wooden tray. |
| strong | pbhs-composition_layout-050 | 2 | -0.104 | +0.125 | **+0.010** | A teapot to the left of three teacups on a small wooden tray. |
| strong | pbhs-composition_layout-004 | 1 | +0.208 | -0.208 | **+0.000** | A beach umbrella to the left of a beach chair on the sand. |
| strong | pbhs-composition_layout-038 | 2 | +0.000 | +0.000 | **+0.000** | A cake placed between two wrapped presents on a wooden table. |
| full | pbh-composition_layout-020 | 2 | +0.021 | -0.042 | **-0.010** | A chess set with the king in the center and two rooks beside it. |
| strong | pbhs-composition_layout-016 | 2 | -0.042 | +0.021 | **-0.010** | A red apple to the left of a green apple on a wooden table. |
| strong | pbhs-composition_layout-070 | 2 | +0.021 | -0.042 | **-0.010** | A tall lamp placed between two short lamps on a long sofa table. |
| strong | pbhs-composition_layout-032 | 2 | -0.042 | +0.021 | **-0.010** | A red kite flying to the left of a blue kite in a clear sky. |
| strong | pbhs-composition_layout-035 | 1 | -0.042 | +0.000 | **-0.021** | A laptop placed to the right of a desk lamp on an office desk. |
| strong | pbhs-composition_layout-058 | 2 | +0.000 | -0.042 | **-0.021** | A deer standing between two trees in a quiet forest clearing. |
| strong | pbhs-composition_layout-062 | 2 | -0.104 | +0.062 | **-0.021** | A flower placed between two leaves on a wooden table. |
| strong | pbhs-composition_layout-025 | 1 | -0.042 | +0.000 | **-0.021** | A salt shaker to the left of a pepper shaker on a dining table. |
| full | pbh-composition_layout-021 | 2 | +0.021 | -0.063 | **-0.021** | Three books stacked vertically with a pair of glasses resting on top. |
| strong | pbhs-composition_layout-044 | 2 | -0.021 | -0.021 | **-0.021** | A black hat to the left of a red scarf on a wooden clothes rack. |
| strong | pbhs-composition_layout-029 | 2 | -0.063 | +0.021 | **-0.021** | A flower vase between two candle holders on a long dining table. |
| full | pbh-composition_layout-016 | 2 | -0.062 | +0.000 | **-0.031** | A vase placed between two lit candles on a wooden mantel. |
| full | pbh-composition_layout-033 | 2 | +0.000 | -0.062 | **-0.031** | Two chairs facing each other across a small dining table. |
| strong | pbhs-composition_layout-021 | 2 | +0.083 | -0.146 | **-0.031** | A pizza placed between two glasses of soda on a wooden table. |
| full | pbh-composition_layout-014 | 2 | -0.042 | -0.042 | **-0.042** | A cat sitting on top of a dresser with a dog sleeping beneath it. |
| strong | pbhs-composition_layout-047 | 1 | -0.042 | -0.042 | **-0.042** | A laptop to the left of an external monitor on a wooden office desk. |
| full | pbh-composition_layout-029 | 2 | -0.021 | -0.062 | **-0.042** | A knife above a wooden cutting board with vegetables arranged beside it. |
| strong | pbhs-composition_layout-036 | 2 | -0.146 | +0.062 | **-0.042** | A cat sitting between two dogs on a striped rug. |
| strong | pbhs-composition_layout-059 | 1 | +0.083 | -0.167 | **-0.042** | A girl standing to the right of a boy in a small classroom. |
| strong | pbhs-composition_layout-060 | 2 | -0.042 | -0.042 | **-0.042** | A castle painted between two tall trees in a fantasy scene. |
| strong | pbhs-composition_layout-057 | 1 | -0.167 | +0.083 | **-0.042** | A guitar to the left of a piano in a small music room. |
| full | pbh-composition_layout-009 | 3 | -0.097 | +0.000 | **-0.049** | A red ball above a blue cube next to a yellow pyramid. |
| full | pbh-composition_layout-001 | 3 | -0.097 | -0.014 | **-0.056** | Two cats sitting on a wooden bench with one bird perched above them. |
| strong | pbhs-composition_layout-049 | 2 | -0.021 | -0.104 | **-0.062** | A red ball to the right of a blue ball on a wooden floor. |
| strong | pbhs-composition_layout-001 | 2 | +0.062 | -0.188 | **-0.062** | A red bowl to the left of a blue bowl on a kitchen counter. |
| strong | pbhs-composition_layout-015 | 2 | -0.062 | -0.063 | **-0.062** | A sunflower placed between two roses in a single glass vase. |
| strong | pbhs-composition_layout-031 | 1 | +0.000 | -0.125 | **-0.062** | A bicycle parked to the right of a scooter on a brick sidewalk. |
| strong | pbhs-composition_layout-040 | 2 | -0.042 | -0.083 | **-0.062** | A vase placed between three picture frames on a long mantel. |
| strong | pbhs-composition_layout-051 | 1 | -0.125 | +0.000 | **-0.062** | A canoe placed to the right of a kayak on a sandy lake shore. |
| strong | pbhs-composition_layout-045 | 2 | -0.021 | -0.125 | **-0.073** | A clock placed between two photographs on a plain wall. |
| full | pbh-composition_layout-003 | 2 | -0.146 | +0.000 | **-0.073** | Three books stacked vertically with a coffee cup on top. |
| strong | pbhs-composition_layout-003 | 2 | -0.062 | -0.104 | **-0.083** | A book placed between two bookends on a wooden shelf. |
| strong | pbhs-composition_layout-028 | 1 | -0.125 | -0.042 | **-0.083** | A book placed between a lamp and a clock on a nightstand. |
| strong | pbhs-composition_layout-042 | 1 | +0.000 | -0.167 | **-0.083** | A pumpkin placed to the right of a haystack in an autumn field. |
| strong | pbhs-composition_layout-030 | 1 | -0.083 | -0.083 | **-0.083** | A sandwich to the left of a bowl of soup on a serving tray. |
| full | pbh-composition_layout-035 | 2 | -0.146 | -0.042 | **-0.094** | A vase of flowers placed exactly between two framed photographs on a mantel. |
| full | pbh-composition_layout-011 | 2 | -0.125 | -0.083 | **-0.104** | Two clocks on a wall, one above the other. |
| full | pbh-composition_layout-015 | 2 | -0.083 | -0.125 | **-0.104** | Four mugs in a row on a kitchen counter with a sugar bowl in the middle. |
| strong | pbhs-composition_layout-006 | 1 | -0.042 | -0.167 | **-0.104** | A tall building between two short buildings in a city skyline. |
| strong | pbhs-composition_layout-066 | 1 | -0.042 | -0.167 | **-0.104** | A piano to the left of a harp on a concert stage. |
| strong | pbhs-composition_layout-026 | 1 | -0.250 | +0.000 | **-0.125** | A child standing to the left of a parent on a sidewalk. |
| strong | pbhs-composition_layout-037 | 1 | -0.125 | -0.125 | **-0.125** | A piggy bank between two stacks of coins on a wooden shelf. |
| strong | pbhs-composition_layout-022 | 2 | -0.083 | -0.167 | **-0.125** | A vase of flowers between two framed pictures on a long wooden mantel. |
| strong | pbhs-composition_layout-012 | 2 | -0.146 | -0.125 | **-0.135** | A pencil lying between two erasers on a wooden desk. |
| strong | pbhs-composition_layout-023 | 1 | -0.208 | -0.083 | **-0.146** | A backpack to the left of a lunchbox on a school desk. |
| strong | pbhs-composition_layout-064 | 1 | -0.333 | +0.042 | **-0.146** | A wooden chair to the right of a small wooden desk. |
| full | pbh-composition_layout-030 | 2 | -0.125 | -0.167 | **-0.146** | A lamp on the left side of a bed with an alarm clock on the right. |
| strong | pbhs-composition_layout-017 | 1 | -0.292 | -0.042 | **-0.167** | A laptop between a notebook and a coffee mug on an office desk. |
| full | pbh-composition_layout-031 | 2 | -0.125 | -0.229 | **-0.177** | A teapot in the center of a tray with two teacups arranged on either side. |
| strong | pbhs-composition_layout-056 | 1 | +0.000 | -0.417 | **-0.208** | A single book between two stacks of books on a study desk. |
| strong | pbhs-composition_layout-027 | 1 | -0.083 | -0.375 | **-0.229** | A mug placed to the right of a kettle on a kitchen counter. |
| strong | pbhs-composition_layout-019 | 2 | -0.271 | -0.208 | **-0.240** | A red tulip to the right of a yellow tulip in a flower bed. |
