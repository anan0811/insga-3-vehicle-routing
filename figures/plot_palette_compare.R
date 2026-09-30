library(ggplot2)
library(dplyr)
library(scales)
library(grid)
library(gridExtra)

csv_path <- "C:/Users/admin/Desktop/My-paper--main/論文/實驗/Case test/TABLE13_FULL_INSGA3.CSV"
out_dir  <- "C:/Users/admin/Desktop/My-paper--main/論文/figures"

df_raw <- read.csv(csv_path, stringsAsFactors = FALSE)

df_wide <- df_raw %>%
  select(day, n_sat, cost_avg, is_optimal) %>%
  mutate(n_sat = factor(n_sat, levels = c(1,2,3),
                        labels = c("Fixed Sat=1","Fixed Sat=2","Fixed Sat=3")))

df_dyn <- df_raw %>%
  filter(is_optimal == 1) %>%
  mutate(opt_label = factor(n_sat, levels = c(1,2,3),
                            labels = c("Optimal = Sat 1","Optimal = Sat 2","Optimal = Sat 3")))

palettes <- list(
  "Okabe-Ito"            = c("#0096FF", "#FF6600", "#00C46A"),
  "ColorBrewer Dark2"    = c("#00CC88", "#FF6B1A", "#8B7FFF"),
  "ColorBrewer Set1"     = c("#2288FF", "#FF2222", "#22CC44")
)

base_theme <- theme_bw(base_size = 10) +
  theme(
    panel.grid.major.x = element_blank(),
    panel.grid.minor   = element_blank(),
    panel.grid.major.y = element_line(linetype = "dotted", colour = "grey75"),
    axis.text.x  = element_text(size = 7),
    axis.text.y  = element_text(size = 8),
    legend.position  = "right",
    legend.key.width = unit(1.5, "cm"),
    legend.text      = element_text(size = 8),
    legend.title     = element_blank(),
    plot.title       = element_text(size = 10, face = "bold", hjust = 0.5),
    plot.margin      = margin(6, 6, 6, 6)
  )

make_panel <- function(pal_name, cols) {
  c1 <- cols[1]; c2 <- cols[2]; c3 <- cols[3]

  cl2 <- c("Fixed Sat=1" = c1, "Fixed Sat=2" = c2, "Fixed Sat=3" = c3)
  lt2 <- c("Fixed Sat=1" = "dashed", "Fixed Sat=2" = "dotted", "Fixed Sat=3" = "longdash")
  pt_cols   <- c("Optimal = Sat 1" = c1, "Optimal = Sat 2" = c2, "Optimal = Sat 3" = c3)
  pt_shapes <- c("Optimal = Sat 1" = 16, "Optimal = Sat 2" = 17, "Optimal = Sat 3" = 15)

  all_cols <- c(cl2,  pt_cols)
  all_lt   <- c(lt2,  c("Optimal = Sat 1"="solid","Optimal = Sat 2"="solid","Optimal = Sat 3"="solid"))

  ggplot() +
    geom_line(data = df_wide,
              aes(x = day, y = cost_avg, colour = n_sat, linetype = n_sat),
              linewidth = 0.75, alpha = 0.85) +
    geom_line(data = df_dyn,
              aes(x = day, y = cost_avg),
              colour = "black", linewidth = 1.1, linetype = "solid") +
    geom_point(data = df_dyn,
               aes(x = day, y = cost_avg, colour = opt_label, shape = opt_label),
               size = 2.6, stroke = 0.3) +
    scale_colour_manual(values = all_cols,
                        breaks = c("Fixed Sat=1","Fixed Sat=2","Fixed Sat=3",
                                   "Optimal = Sat 1","Optimal = Sat 2","Optimal = Sat 3")) +
    scale_linetype_manual(values = lt2,
                          breaks = c("Fixed Sat=1","Fixed Sat=2","Fixed Sat=3")) +
    scale_shape_manual(values = pt_shapes,
                       breaks = c("Optimal = Sat 1","Optimal = Sat 2","Optimal = Sat 3")) +
    scale_x_continuous(breaks = 1:30, expand = expansion(add = 0.5)) +
    scale_y_continuous(labels = comma) +
    labs(x = "Planning Day", y = "Total Cost ($)",
         title = pal_name,
         colour = NULL, linetype = NULL, shape = NULL) +
    guides(
      colour   = guide_legend(order = 1, override.aes = list(linewidth = 0.8, shape = NA)),
      linetype = guide_legend(order = 1),
      shape    = guide_legend(order = 2,
                              override.aes = list(colour = c(c1, c2, c3), size = 2.5))
    ) +
    base_theme
}

plots <- mapply(make_panel, names(palettes), palettes, SIMPLIFY = FALSE)

out <- file.path(out_dir, "palette_compare.pdf")
pdf(out, width = 11, height = 12)
grid.arrange(grobs = plots, ncol = 1)
dev.off()
cat("Saved:", out, "\n")
