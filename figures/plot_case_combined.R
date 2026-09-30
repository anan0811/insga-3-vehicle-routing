library(ggplot2)
library(dplyr)
library(scales)
library(patchwork)

csv_path <- "C:/Users/admin/Desktop/My-paper--main/論文/實驗/Case test/TABLE13_FULL_INSGA3.CSV"
out_dir  <- "C:/Users/admin/Desktop/My-paper--main/論文/figures"

df_raw <- read.csv(csv_path, stringsAsFactors = FALSE)

df_wide <- df_raw %>%
  select(day, n_sat, cost_avg, nv_avg, co2_avg, is_optimal) %>%
  mutate(n_sat = factor(n_sat, levels = c(1,2,3),
                        labels = c("Fixed Sat=1","Fixed Sat=2","Fixed Sat=3")))

df_dyn <- df_raw %>%
  filter(is_optimal == 1) %>%
  arrange(day) %>%
  mutate(opt_label = factor(n_sat, levels = c(1,2,3),
                            labels = c("Optimal = Sat 1","Optimal = Sat 2","Optimal = Sat 3")))

# colours & shapes
c1 <- "#2288FF"; c2 <- "#FF6600"; c3 <- "#22CC44"
line_cols  <- c("Fixed Sat=1" = c1, "Fixed Sat=2" = c2, "Fixed Sat=3" = c3)
line_types <- c("Fixed Sat=1" = "dashed", "Fixed Sat=2" = "dotted", "Fixed Sat=3" = "longdash")
pt_shapes  <- c("Optimal = Sat 1" = 16, "Optimal = Sat 2" = 17, "Optimal = Sat 3" = 15)
pt_cols    <- c("Optimal = Sat 1" = c1, "Optimal = Sat 2" = c2, "Optimal = Sat 3" = c3)
all_cols   <- c(line_cols, pt_cols)

base_theme <- theme_bw(base_size = 9) +
  theme(
    panel.grid.major.x = element_blank(),
    panel.grid.minor   = element_blank(),
    panel.grid.major.y = element_line(linetype = "dotted", colour = "grey75"),
    axis.text.x  = element_text(size = 6.5),
    axis.text.y  = element_text(size = 7.5),
    legend.text  = element_text(size = 8),
    legend.title = element_blank(),
    legend.key.width = unit(1.4, "cm"),
    plot.title   = element_text(size = 9.5, face = "bold", hjust = 0.5),
    plot.margin  = margin(4, 4, 4, 4)
  )

make_panel <- function(col, y_label, panel_title, y_fmt = comma) {
  ggplot() +
    geom_line(data = df_wide,
              aes(x = day, y = .data[[col]], colour = n_sat, linetype = n_sat),
              linewidth = 0.65, alpha = 0.85) +
    geom_line(data = df_dyn,
              aes(x = day, y = .data[[col]]),
              colour = "black", linewidth = 1.0, linetype = "solid") +
    geom_point(data = df_dyn,
               aes(x = day, y = .data[[col]], colour = opt_label, shape = opt_label),
               size = 2.2, stroke = 0.3) +
    scale_colour_manual(values = all_cols,
                        breaks = c("Fixed Sat=1","Fixed Sat=2","Fixed Sat=3",
                                   "Optimal = Sat 1","Optimal = Sat 2","Optimal = Sat 3")) +
    scale_linetype_manual(values = line_types,
                          breaks = c("Fixed Sat=1","Fixed Sat=2","Fixed Sat=3")) +
    scale_shape_manual(values = pt_shapes,
                       breaks = c("Optimal = Sat 1","Optimal = Sat 2","Optimal = Sat 3")) +
    scale_x_continuous(breaks = seq(1,30,2), expand = expansion(add = 0.5)) +
    scale_y_continuous(labels = y_fmt) +
    labs(x = "Planning Day", y = y_label, title = panel_title,
         colour = NULL, linetype = NULL, shape = NULL) +
    guides(
      colour = guide_legend(
        override.aes = list(
          linetype  = c("dashed","dotted","longdash","blank","blank","blank"),
          shape     = c(NA, NA, NA, 16, 17, 15),
          linewidth = c(0.8, 0.8, 0.8, 0, 0, 0),
          size      = c(NA, NA, NA, 2.5, 2.5, 2.5)
        )
      ),
      linetype = "none",
      shape    = "none"
    ) +
    base_theme
}

p_cost <- make_panel("cost_avg", "Total Cost ($)",       "(a) Cost",     comma)
p_nv   <- make_panel("nv_avg",   "Number of Vehicles",   "(b) NV",
                     number_format(accuracy = 0.1))
p_co2  <- make_panel("co2_avg",  expression(CO[2]~(kg)), "(c) CO₂", comma)

combined <- (p_cost | p_nv | p_co2) +
  plot_layout(guides = "collect") &
  theme(legend.position = "bottom",
        legend.box = "horizontal")

out <- file.path(out_dir, "case_trends_combined_r.pdf")
ggsave(out, plot = combined, device = "pdf", width = 14, height = 5, units = "in")
cat("Saved:", out, "\n")
